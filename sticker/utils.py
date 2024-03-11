import sys
import os
import json
from copy import copy
import imageio
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs.config_manager import get_config_value as gcv

###################################
# Metadata.json utility functions #
###################################

def get_metadata(chunk_no) -> dict:
    # Check to see if an mask metadata file exists
    part_name = f"part-{str(chunk_no).zfill(6)}"

    mask_path = os.path.join(gcv("DATA_PATH"), "mask", part_name, f"{part_name}.json")
    if os.path.exists(mask_path):
        with open(mask_path, "r") as f:
            metadata = json.load(f)
        return metadata
    else:
        image_path = os.path.join(gcv("DATA_PATH"), "image", part_name, f"{part_name}.json")
        with open(image_path, "r") as f:
            metadata = json.load(f)
        return metadata

def save_metadata(metadata, chunk_no):
    part_name = f"part-{str(chunk_no).zfill(6)}"
    path = os.path.join(gcv("DATA_PATH"), "mask", part_name, f"{part_name}.json")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)

def save_segmented_data(segmented_data, metadata, image_name, save_root):
    metadata = copy(metadata)

    metadata[image_name]["seg_data"] = dict()

    for i, data in enumerate(segmented_data):
        if gcv("SAVE_ANNOTATED_IMAGES"):
            _, logits, phrase, mask, annotated_image = data
        else:
            _, logits, phrase, mask = data

        metadata[image_name]["seg_data"][i] = {
            "logits": float(logits),
            "phrase": phrase,
        }

        # Save the annotated image if we are saving them
        if gcv("SAVE_ANNOTATED_IMAGES"):
            annotated_image_path = os.path.join(save_root, f"{image_name.replace('.png', '')}_{str(i).zfill(4)}_annotated.png")
            annotated_image.save(annotated_image_path)

        # Save the mask
        mask_path = os.path.join(save_root, f"{image_name.replace('.png', '')}_{str(i).zfill(4)}_mask.png")
        uint8_mask_array = (mask * 255).astype(np.uint8)
        imageio.imwrite(mask_path, uint8_mask_array)

    return metadata

###################################
# Image utility functions         #
###################################

def load_image(image_id : str, chunk : int):
    if image_id.endswith(".png"):
        image_id = image_id.replace(".png", "")

    part_name = f"part-{str(chunk).zfill(6)}"
    image_path = os.path.join(gcv("DATA_PATH"), "image", part_name, f"{image_id}.png")
    return imageio.imread(image_path)

def load_segmentation(image_id, seg_id, chunk) -> np.ndarray:
    if image_id.endswith(".png"):
        image_id = image_id.replace(".png", "")

    part_name = f"part-{str(chunk).zfill(6)}"
    mask_path = os.path.join(gcv("DATA_PATH"), "mask", part_name, f"{image_id.replace('.png', '')}_{str(seg_id).zfill(4)}_mask.png")
    mask = imageio.imread(mask_path)
    mask = np.array(mask / 255).astype(np.uint8)
    return mask

def create_image_cutout(image : np.ndarray, mask : np.ndarray) -> np.ndarray:
    cutout = copy(image)
    cutout[mask == 0] = 0
    return cutout