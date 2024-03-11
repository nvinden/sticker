
import sys
import os
import spacy

sys.path.append("/home/vision/projects/sticker/packages/GroundingDINO")
sys.path.append("/home/vision/projects/sticker/packages/segment-anything")

from PIL import Image

import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict, load_model

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline

from huggingface_hub import hf_hub_download

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs.config_manager import get_config_value as gcv

DEVICE = "cuda"

# If you have multiple GPUs, you can set the GPU to use here.
# The default is to use the first GPU, which is usually GPU 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seg_image(image_path, phrase_list, gd_model : torch.nn.Module, sam_model : SamPredictor):
    # Create (boxes, logits, phrases, mask, annotated_image) for each phrase
    bouding_boxes_data = extract_bounding_boxes(image_path, phrase_list, gd_model)[:100]
    final_data = create_sam_masks(image_path, bouding_boxes_data, sam_model, return_seg_image=gcv("SAVE_ANNOTATED_IMAGES"))

    return final_data

def create_sam_masks(image_path, bouding_boxes, sam_predictor : SamPredictor, return_seg_image=False):
    image_source, _ = load_image(image_path)
    sam_predictor.set_image(image_source)

    masks_list = []

    for i, (boxes, logits, phrases) in enumerate(bouding_boxes):
        curr_data = [boxes.cpu().numpy(), float(logits), phrases]

        if len(boxes) == 0:
            #print(f"Skipping phrase {i} because no boxes were found.")
            continue

        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
        masks, _, _ = sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
        
        # Always one bounding box, so always one mask
        assert list(masks.shape) == [1, 1, H, W]

        single_mask = masks[0][0]
        single_mask = single_mask.cpu().numpy()
        single_mask = curate_mask(single_mask)

        if single_mask is None:
            continue

        curr_data.append(single_mask)

        if return_seg_image:
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=[phrases])
            annotated_frame = annotated_frame[...,::-1] # BGR to RGB

            annotated_frame_with_mask = show_mask(single_mask, annotated_frame, random_color = False)

            annotated_image = Image.fromarray(annotated_frame_with_mask)

            curr_data.append(annotated_image)

        masks_list.append(tuple(curr_data))

    return masks_list

# Inputs a mask and smooths it out, fills in holes finalizes it
# If the mask is in too many pieces, it will return None
def curate_mask(mask: np.ndarray):
    # Ensure the mask is in uint8 format
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Smooth the mask using morphological closing and Gaussian blur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_blurred = cv2.GaussianBlur(mask_closed, (7, 7), 0)
    _, smoothed_mask = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
    
    # Check if the mask is in too many pieces
    num_labels, _ = cv2.connectedComponents(smoothed_mask)
    if num_labels - 1 != 1:  # More than one object piece
        return None

    # Check if the mask is at least 10% of the total image size
    mask_area = np.count_nonzero(smoothed_mask)
    total_area = mask.shape[0] * mask.shape[1]
    if mask_area < total_area * 0.10:
        return None

    # Convert filled mask back to boolean
    final_mask = smoothed_mask.astype(bool)
    
    return final_mask


def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([255/255, 0/255, 0/255, 0.3])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

# Generating all the bounding boxes for each phrase
def extract_bounding_boxes(image_path, phrase_list, gd_model : torch.nn.Module):
    _, image = load_image(image_path)

    boxes_list = []

    for i, phrase in enumerate(phrase_list):
        # TODO: use own custon function to allow for only one image use, and
        #      to allow for better thresholding
        boxes, logits, refactored_phrases = predict(
            model=gd_model, 
            image=image, 
            caption=phrase, 
            box_threshold=gcv("BOX_THRESHOLD"), 
            text_threshold=gcv("TEXT_THRESHOLD"),
        )

        # Add each box, logit, and phrase to the list induvidually
        for j in range(len(boxes)):
            refactored_phrase = refactored_phrases[j].replace("\n", " ").replace("[CLS]", "").replace("[SEP]", "")
            boxes_list.append((boxes[j].unsqueeze(0), logits[j].unsqueeze(0), refactored_phrase))

    return boxes_list

# Load grounding DINO model
def build_gd_model(filename, ckpt_config_filename) -> torch.nn.Module:
    gd_model = load_model(ckpt_config_filename, filename)
    gd_model.to(device=DEVICE)
    return gd_model

def build_sam_model(filename) -> SamPredictor:
    sam = build_sam(checkpoint=filename)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

