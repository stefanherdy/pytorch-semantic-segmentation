#!/usr/bin/env python3

import yaml
import requests
import cv2
import numpy as np
import ndjson
import os

# Helper function to transform cathegorical image to colour image
def logits2rgb(img):
    # Specify custom colours
    red = [200, 0, 10]
    green = [187,207, 74]
    blue = [0,108,132]
    yellow = [255,204,184]
    black = [0,0,0]
    white = [226,232,228]
    cyan = [174,214,220]
    orange = [232,167,53]

    colours = [red, green, blue, yellow, black, white, cyan, orange]

    
    
    shape = np.shape(img)
    h = int(shape[0])
    w = int(shape[1])
    col = np.zeros((h, w, 3))
    unique = np.unique(img)
    for i, val in enumerate(unique):
        mask = np.where(img == val)
        for j, row in enumerate(mask[0]):
            x = mask[0][j]
            y = mask[1][j]
            col[x, y, :] = colours[int(val)]

    return col.astype(int)

def get_mask(PROJECT_ID, api_key, colour, class_indices):
    # Open export json. Change name if required
    with open('./export-result.ndjson') as f:
        data = ndjson.load(f)
        # Iterate over all images
        for i, d in enumerate(data):
            files_in_folder = os.listdir('./labels_cathegorical/')
            image_name = data[i]['data_row']['external_id']
            label_name = image_name.replace(".JPG", "") + '-mask.png'
            if label_name not in files_in_folder:
                mask_full = np.zeros((data[i]['media_attributes']['height'], data[i]['media_attributes']['width']))
                # Iterate over all masks
                for idx, obj in enumerate(data[i]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects']):
                    # Extract mask name and mask url
                    name = data[i]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects'][idx]['name']
                    url = data[i]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects'][idx]['mask']['url']

                    cl = class_indices[name]
                    print('Class ' + name + ' assigned to class index ' + str(cl))
                    
                    # Download mask
                    headers = {'Authorization': api_key}
                    with requests.get(url, headers=headers, stream=True) as r:
                        r.raw.decode_content = True
                        mask = r.raw
                        image = np.asarray(bytearray(mask.read()), dtype="uint8")
                        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                    # Assign mask index to image-mask 
                    mask = np.where(image == 255)
                    mask_full[mask] = cl

                unique = np.unique(mask_full)
                print('The masks of the image are: ')
                print(unique)
                if len(unique) > 1:
                    if colour == True:
                        mask_full_colour = logits2rgb(mask_full)
                        mask_full_colour = cv2.cvtColor(mask_full_colour.astype('float32'), cv2.COLOR_RGB2BGR)
                    #res_mask = cv2.resize(mask_full, (512, 512), interpolation = cv2.INTER_NEAREST)
                    # Save Image
                    cv2.imwrite('./labels_colour/' + image_name.replace(".JPG", "") + '-mask.png', mask_full_colour)
                cv2.imwrite('./labels_cathegorical/' + image_name.replace(".JPG", "") + '-mask.png', mask_full)
            else:
                print('File "' + label_name + '" already processed!')
        print('')



if __name__ == "__main__":
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    project_id = config['project_id']
    api_key = config['api_key']
    colour = True

    class_indices = {	"lichen" : 1,
                        "cyano pale" : 2,
                        "cyano dark" : 3,
                        "vascular_plants" : 4,
                        "moss" : 5,
                        "other" : 6,
                        }

    get_mask(project_id, api_key, colour, class_indices)
