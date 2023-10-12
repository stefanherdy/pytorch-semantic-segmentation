import os
import shutil
import cv2
import numpy as np

def crop_bg(mask_cat, mask_colour, raw_image):
    # Define the annotation values (non-background values)

    # Find the non-background coordinates in the mask image
    #annotation_values = [1, 2, 3, 4, 5, 6]
    #non_bg_coords = np.argwhere(np.isin(mask_cat, annotation_values))
    non_bg_coords = np.argwhere(mask_cat > 0)
    # Get the minimum and maximum coordinates to find the bounding box
    upper_left = np.min(non_bg_coords, axis=0)
    lower_right = np.max(non_bg_coords, axis=0)

    # Crop the mask and raw image
    cropped_mask_cat = mask_cat[upper_left[0]:lower_right[0] + 1, upper_left[1]:lower_right[1] + 1]
    cropped_mask_colour = mask_colour[upper_left[0]:lower_right[0] + 1, upper_left[1]:lower_right[1] + 1]
    cropped_raw_image = raw_image[upper_left[0]:lower_right[0] + 1, upper_left[1]:lower_right[1] + 1]

    return cropped_mask_cat, cropped_mask_colour, cropped_raw_image


# Define the paths to your folders
label_cat_folder = './labels_cathegorical'
label_colour_folder = './labels_colour'
raw_data_folder = './data/all'
cropped_raw_data_folder = './raw_data'
cropped_label_cat_folder = './mask_cat_cropped'
cropped_label_colour_folder = './mask_colour_cropped'


# Ensure that folder 3 exists
if not os.path.exists(cropped_raw_data_folder):
    os.makedirs(cropped_raw_data_folder)

# Iterate through the files in folder 1
for filename in os.listdir(label_cat_folder):
    # Check if the file is an image (you can extend this check as needed)
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        filename_raw = filename[:-9] + '.JPG'
        # Create the source and destination paths
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

        # Check if the image with the same name exists in folder 2
        if os.path.isfile(raw_data_path):
            # Copy the image from folder 2 to folder 3
            cv2.imwrite(mask_cat_destination_path, cropped_mask_cat)
            cv2.imwrite(mask_colour_destination_path, cropped_mask_colour)
            cv2.imwrite(raw_data_destination_path, cropped_raw_image)
            #shutil.copy(source_path, )
            print(f"Copied: {filename_raw} from {raw_data_folder} to {cropped_raw_data_folder}")
        else:
            print(f"Image {filename_raw} not found in {raw_data_folder}")



