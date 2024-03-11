import sys
import os
import spacy
import json
from copy import copy
from tqdm import tqdm
import imageio
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs.config_manager import set_config, print_config
from configs.config_manager import get_config_value as gcv
from sticker.data import SDRawDataset
from sticker.nlp import extract_phrases, extract_phrases_hierarchical
from sticker.segmentation import seg_image, build_gd_model, build_sam_model
from sticker.utils import save_metadata, get_metadata, save_segmented_data

CHUNK_NOS = list(range(1, 2 + 1))

###################################
# NLP utility functions           #
###################################

# Takes metadata and for each row, splits the sentences into noun chunks
# Saves them in the metadata as metadata["chunks"] = list(str)
def precompute_nlp(metadata, nlp_mod):
    for id, row in metadata.items():
        noun_chunks = extract_phrases_hierarchical(row["p"], nlp_mod)
        metadata[id]["all_chunks"] = noun_chunks

    return metadata

def is_metadata_nlp_computed(metadata : dict):
    for id, row in metadata.items():
        if "all_chunks" not in row:
            return False
    return True

def segment():
    # Loading models: Hierarchical Object Parsing, groundingDINO, and SegmentAnything
    nlp_mod = spacy.load("en_core_web_trf")
    gd_mod = build_gd_model(filename="saves/gd/groundingdino_swint_ogc.pth", ckpt_config_filename="saves/gd/GroundingDINO_SwinT_OGC.py")
    sam_mod = build_sam_model(filename="saves/sam/sam_vit_h_4b8939.pth")

    # Load data from each chunk
    for chunk_no in CHUNK_NOS:
        print(f"Processing chunk {chunk_no}...")

        data_module = SDRawDataset(chunk_no)

        mask_metadata = get_metadata(chunk_no)

        # Precompute NLP if it is not already
        if not is_metadata_nlp_computed(mask_metadata):
            mask_metadata = precompute_nlp(mask_metadata, nlp_mod)

            # Save the metadata
            save_metadata(mask_metadata, chunk_no)

        start_time = time.time()

        # For each image in the chunk, segment the image and save the data
        for i, data in enumerate(tqdm(data_module, desc=f"Processing chunk {chunk_no}")):

            # Checks if the image has already been segmented and saved:
            if mask_metadata[data["image_name"]].get("seg_data") is not None:
                continue

            # Root to save the data in
            save_root = os.path.join(gcv("DATA_PATH"), "mask", f"part-{str(chunk_no).zfill(6)}")

            noun_chunks = mask_metadata[data["image_name"]]["all_chunks"]

            # segmented_data:
            # (boxes, logits, phrases, mask, annotated_image) for each phrase
            segmented_data = seg_image(data["image_path"], noun_chunks, gd_mod, sam_mod)

            # Save the segmented data
            mask_metadata = save_segmented_data(segmented_data, mask_metadata, data["image_name"], save_root)

            # Save the metadata
            save_metadata(mask_metadata, chunk_no)

        print(f"Chunk {chunk_no} done!")
        print("Total time: ", time.time() - start_time)

    print("Done!")


if __name__ == '__main__':
    # Set the configuration
    set_config("base")
    print_config()

    # Run the segmentation
    segment()
