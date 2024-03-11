import os
import sys
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
from PIL import Image
import json
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs.config_manager import get_config_value as gcv

################################
# Sticker Segmentation Dataset #
################################

N_DIFFUSION_FILES = 2000

class SDRawDataset(Dataset):
    def __init__(self, split, return_mode = "real"):
        """
        Args:
            data: Data to load and prepare. This could be a path to a file, a list of files, etc.
            transforms: Optional transform to be applied on a sample.
        """
        self.split = split
        self.return_mode = return_mode

        assert self.return_mode in ["real", "tensor"], "Invalid return mode. Must be one of 'real' or 'tensor'."

        included_data = def_get_ttv_df(self.split)

        self.img_desc = self.filter_image_descriptions(included_data)
        self.img_desc = self.img_desc.to_dict(orient="records")

        self.image_paths = self.get_image_paths()

    def __len__(self):
        return len(self.img_desc)

    def __getitem__(self, idx):
        """
        Args:
            idx: Index of the data to load and return.

        Returns:
            sample: The data sample at index `idx`.
        """
        id_row = self.img_desc[idx]
        image = self.load_image(id_row)

        if self.return_mode == "real":
            sample = {
                "image": image,
                "image_path": self.image_paths[id_row["img_no"]],
                "part_id": id_row["part_id"],
                "image_name": id_row["image_name"],
                "prompt": id_row["prompt"],
                "width": id_row["width"],
                "height": id_row["height"]
            }
        else:
            pass # TODO: Implement tensor mode
            

        return sample
    
    # take in a dataframe and filter them for no NSFW files
    # also removes unneeded columns
    # also adds a img_no column
    # remaining columns: "img_no", "image_id", "prompt", "image_description", "width", "height"
    def filter_image_descriptions(self, df : pd.DataFrame):
        df =  df[df["image_nsfw"] <= 0.3]
        df = df[df["prompt_nsfw"] <= 0.15]
        df = df.drop(columns=["image_nsfw", "prompt_nsfw", "seed", "step", "cfg", "sampler", "user_name", "timestamp"])
        df["img_no"] = range(0, len(df))

        return df
    
    # Precomputes all of the image paths for the dataset
    def get_image_paths(self):
        base_path = gcv("BASE_PATH")
        data_path = gcv("DATA_PATH")

        image_paths = []
        for i in range(len(self.img_desc)):
            curr_image_path = os.path.join(base_path, data_path, "image", f"part-{str(self.img_desc[i]['part_id']).zfill(6)}", f"{self.img_desc[i]['image_name']}")
            image_paths.append(curr_image_path)
        return image_paths
    
    # Loading the image from the file path
    def load_image(self, image_desc_row):
        img_path = self.image_paths[image_desc_row["img_no"]]
        img = Image.open(img_path)

        return img

# Helper functions
def def_get_ttv_df(chunks):
    mode = gcv("RUN_MODE")
    data_path = gcv("DATA_PATH")

    files = []

    if isinstance(chunks, int):
        chunks = [chunks]

    if mode == "debug":
        metadata_path = os.path.join(data_path, "image", "metadata-debug.csv")
    else:
        metadata_path = os.path.join(data_path, "image", "metadata.csv")
    metadata = pd.read_csv(metadata_path)

    if "Unnamed: 0" in metadata.columns:
        metadata = metadata.drop('Unnamed: 0', axis=1)
    
    df = metadata[metadata["part_id"].isin(chunks)]

    return df


#######################################
# Sticker Generation Training Dataset #
#######################################

class ControlNetDataset(Dataset):
    def __init__(self, folds):
        self.folds = folds

        self.data = self.get_data_list()

    # Creates a full list of tuples of all the data
    # Each tuple is (image_path, text, hint_path)
    def get_data_list(self):
        data_list = []

        for fold in self.folds:
            part_id = f"part-{str(fold).zfill(6)}"
            mask_metadata_path = os.path.join(gcv("BASE_PATH"), gcv("DATA_PATH"), "mask", part_id, f"{part_id}.json")
            
            with open(mask_metadata_path, 'r') as file:
                mask_metadata = json.load(file)

            # For each base image, get the corresponding masks and texts from the segmentation
            for image_name, metadata in mask_metadata.items():
                # If there is no segmentation data, skip
                if metadata.get("seg_data") is None:
                    continue

                # For each mask-subtext segmentation
                for seg_no, seg_dat in metadata["seg_data"].items():
                    seg_no = int(seg_no)

                    phrase = seg_dat["phrase"]
                    image_path = os.path.join(gcv("BASE_PATH"), gcv("DATA_PATH"), "image", part_id, image_name)
                    mask_path = os.path.join(gcv("BASE_PATH"), gcv("DATA_PATH"), "mask", part_id, f"{image_name.replace('.png', '')}_{str(seg_no).zfill(4)}_mask.png")
                    
                    data_list.append((image_path, phrase, mask_path))

        return data_list

    def __len__(self):
        return len(self.data)

    # Outputs: image, text, hint (mask)
    def __getitem__(self, idx):
        target_path, prompt, hint_path = self.data[idx]

        target = Image.open(target_path)
        hint = Image.open(hint_path)

        # Convert image and hint to numpy
        target = np.array(target)
        hint = np.array(hint)
        hint = np.stack((hint,)*3, axis=-1) # Convert to 3 channel

        target, hint = self.preprocess_image(target, hint)

        # Normalize target images to [0, 1].
        target = target.astype(np.float32) / 255.0

        # Normalize hint images to [-1, 1].
        hint = (hint.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=hint)
    
    # 1) Crops the image to fit the mask
    def preprocess_image(self, image, mask, target_size=(512, 512)):
        # Convert mask to grayscale and then to binary to find contours
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find the bounding box of the mask
        x, y, w, h = cv2.boundingRect(mask_binary)

        # Crop the image and mask
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]

        # Calculate the ratio to resize the image and mask without stretching
        ratio = float(target_size[0]) / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))

        # Resize the cropped regions without changing the aspect ratio
        resized_image = cv2.resize(cropped_image, new_size, interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(cropped_mask, new_size, interpolation=cv2.INTER_AREA)

        # Create a new canvas and center the resized image and mask
        new_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        new_mask = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Calculate the centering position
        x_offset = (target_size[0] - new_size[0]) // 2
        y_offset = (target_size[1] - new_size[1]) // 2

        # Place the resized image and mask into the center of the canvas
        new_image[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized_image
        new_mask[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized_mask

        return new_image, new_mask

