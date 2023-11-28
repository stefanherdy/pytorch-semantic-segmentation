#!/usr/bin/env python3

import os
import shutil
import cv2
import numpy as np

def crop_bg(mask_cat, mask_colour, raw_image):
    # Crop background (categorical value = 0)
    non_bg_coords = np.argwhere(mask_cat > 0)
    upper_left = np.min(non_bg_coords, axis=0)
    lower_right = np.max(non_bg_coords, axis=0)

    cropped_mask_cat = mask_cat[upper_left[0]:lower_right[0] + 1, upper_left[1]:lower_right[1] + 1]
    cropped_mask_colour = mask_colour[upper_left[0]:lower_right[0] + 1, upper_left[1]:lower_right[1] + 1]
    cropped_raw_image = raw_image[upper_left[0]:lower_right[0] + 1, upper_left[1]:lower_right[1] + 1]

    return cropped_mask_cat, cropped_mask_colour, cropped_raw_image


label_cat_folder = './labels_cathegorical'
label_colour_folder = './labels_colour'
raw_data_folder = './data/all'
cropped_raw_data_folder = './raw_data'
cropped_label_cat_folder = './mask_cat_cropped'
cropped_label_colour_folder = './mask_colour_cropped'


if not os.path.exists(cropped_raw_data_folder):
    os.makedirs(cropped_raw_data_folder)

for filename in os.listdir(label_cat_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        filename_raw = filename[:-9] + '.JPG'
        raw_data_path = os.path.join(raw_data_folder, filename_raw)
        raw_data_destination_path = os.path.join(cropped_raw_data_folder, filename_raw)
        mask_cat_path = os.path.join(label_cat_folder, filename)
        mask_colour_path = os.path.join(label_colour_folder, filename)
        mask_cat_destination_path = os.path.join(cropped_label_cat_folder, filename)
        mask_colour_destination_path = os.path.join(cropped_label_colour_folder, filename)

        raw_image = cv2.imread(raw_data_path)
        mask_cat = cv2.imread(mask_cat_path, cv2.IMREAD_GRAYSCALE)
        mask_colour = cv2.imread(mask_colour_path)

        cropped_mask_cat, cropped_mask_colour, cropped_raw_image = crop_bg(mask_cat, mask_colour, raw_image)

        if os.path.isfile(raw_data_path):
            cv2.imwrite(mask_cat_destination_path, cropped_mask_cat)
            cv2.imwrite(mask_colour_destination_path, cropped_mask_colour)
            cv2.imwrite(raw_data_destination_path, cropped_raw_image)
            print(f"Copied: {filename_raw} from {raw_data_folder} to {cropped_raw_data_folder}")
        else:
            print(f"Image {filename_raw} not found in {raw_data_folder}")



