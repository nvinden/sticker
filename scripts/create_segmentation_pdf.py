import sys
import os
import spacy
import json
from copy import copy, deepcopy
from tqdm import tqdm
import imageio
import numpy as np
import time
import io

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs.config_manager import set_config, print_config
from configs.config_manager import get_config_value as gcv
from sticker.data import SDRawDataset
from sticker.nlp import extract_phrases, extract_phrases_hierarchical
from sticker.segmentation import seg_image, build_gd_model, build_sam_model
from sticker.utils import save_metadata, get_metadata, save_segmented_data, load_image, load_segmentation, create_image_cutout

from fpdf import FPDF
from PIL import Image


FOLD_NO = 2

class SegmentationPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font('DejaVu', '', 'packages/DejaVuSansCondensed.ttf', uni=True)
        self.set_font('DejaVu', size=12)
        self.set_auto_page_break(auto=True, margin=15)
        #self.add_page()

        self.section_count = 1

    def add_new_section(self, img_id, row):
        self.add_page()
        self.set_font("DejaVu", size=16)
        self.cell(200, 10, f"{self.section_count}) Image ID: {img_id}", ln=True, align="L")
        self.set_font("DejaVu", size=10)
        self.multi_cell(0, 5, "Prompt: " + row["p"])
        self.ln(5)

        self.section_count += 1

    def numpy_array_to_image_file(self, numpy_array, format='PNG'):
        """
        Converts a NumPy array to a PIL Image and saves it to a bytes buffer.
        Returns the buffer which can be used as a file by FPDF.
        """
        image = Image.fromarray(numpy_array)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=format)
        img_buffer.seek(0)  # Move to the start of the buffer
        return img_buffer

    def add_new_segment(self, base_img, mask_img, cutout_img, phrase, logits):
        page_width = 210 - 20
        img_width = page_width / 3
        img_height = img_width
        
        # Check if adding the new segment would exceed the page's bottom margin
        start_x = 10
        start_y = self.get_y()
        needed_space = img_height + 20  # Estimated space needed for images and text
        
        if (start_y + needed_space) > (297 - 15):  # 297mm is the height of an A4 page, 15mm is the bottom margin
            self.add_page()  # Add a new page if needed
            start_y = self.get_y()  # Update start_y for the new page
        
        # Convert and add images
        base_img_file = self.numpy_array_to_image_file(base_img, format='PNG')
        mask_img_file = self.numpy_array_to_image_file((mask_img * 255).astype(np.uint8), format='PNG')
        cutout_img_file = self.numpy_array_to_image_file(cutout_img, format='PNG')
        
        self.image(base_img_file, start_x, start_y, img_width, img_height)
        self.image(mask_img_file, start_x + img_width, start_y, img_width, img_height)
        self.image(cutout_img_file, start_x + 2 * img_width, start_y, img_width, img_height)
        
        # Move below the images for text
        self.set_y(start_y + img_height + 5)
        
        # Add logit and phrase text
        self.set_font("DejaVu", size=10)
        self.cell(0, 5, f'Logit: {logits}', ln=True)
        self.multi_cell(0, 5, f'Phrase: {phrase}')
        self.ln(5)

def create_row_pages(pdf_c : SegmentationPDF, image_id : str, row : dict):
    pdf_c.add_new_section(image_id, row)

    if "seg_data" not in row:
        return pdf_c
    
    # Sort seg_data by logits, descending
    sorted_seg_data = sorted(row["seg_data"].items(), key=lambda item: item[1]["logits"], reverse=True)
    
    for i, seg_data in sorted_seg_data:
        logits = seg_data["logits"]
        phrase = seg_data["phrase"]

        # Loading the images
        image = load_image(image_id, chunk = FOLD_NO)
        mask = load_segmentation(image_id, i, chunk = FOLD_NO)
        cutout = create_image_cutout(image, mask)

        pdf_c.add_new_segment(image, mask, cutout, phrase, logits)

    return pdf_c

def create_pdf():
    metadata = get_metadata(FOLD_NO)

    pdf_c = SegmentationPDF()

    for i, (id, row) in tqdm(enumerate(metadata.items()), total=len(metadata)):
        pdf_c = create_row_pages(pdf_c, id, row)

        if i % 30 == 0:
            temp_pdf_c = deepcopy(pdf_c)
            temp_pdf_c.output("segmentation.pdf")

    pdf_c.output("segmentation.pdf")

        

if __name__ == "__main__":
    set_config("base")

    create_pdf()