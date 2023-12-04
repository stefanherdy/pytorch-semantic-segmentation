import labelbox
import labelbox.data.annotation_types as lb_types
from PIL import Image
import os
import numpy as np
import yaml
import uuid
from pathlib import Path

# Important!!
# Masks need to have the same name as raw image and must be .png files.

# Specify your project onthology here
CLASS_DICT = {	"lichen" : 1,
                "cyano pale" : 2,
                "cyano dark" : 3,
                "vascular_plants" : 4,
                "moss" : 5,
                "other" : 6,
                }

def export_project_data(project):
    labels = project.export_v2(params={
        "data_row_details": True,
        "metadata_fields": True,
        "attachments": True,
        "project_details": True,
        "performance_details": True,
        "label_details": True,
        "interpolated_frames": True
    })
    return labels

def perform_upload(client, project, img_name, mask):
    labels = []
    annotations = []
    class_names = list(CLASS_DICT.keys())
    class_values = np.unique(mask)
    # Iterate over all classes to merge annotations
    for class_value in class_values:
        color = (class_value, class_value, class_value)

        mask_data = lb_types.MaskData.from_2D_arr(arr=mask)
        mask_annotation = lb_types.ObjectAnnotation(
            name = class_names[class_value-1], # must match your ontology feature"s name
            value=lb_types.Mask(mask=mask_data, color=color),
            )
        annotations.append(mask_annotation)

    labels.append(
        lb_types.Label(data=lb_types.ImageData(global_key=img_name),
                    annotations=annotations))
    
    # Upload
    upload_job = labelbox.MALPredictionImport.create_from_objects(
        client = client, 
        project_id = project.uid, 
        name=f"mal_job{img_name}{uuid.uuid4()}", 
        predictions=labels
    )

    # Check if upload was successful
    if len(upload_job.errors) > 0:
        print(upload_job.errors)
    else:
        print("Upload successful")

def upload_masks(client, project, to_label_list, mask_folder, export_json):
    mask_files = [str(file) for file in Path(mask_folder).glob("*.png")]
    upload_count = 1
    for mask_file in mask_files:
        # Raw images saved as .JPG so we have to change our .png mask extension to extension of raw images.
        img_name = os.path.basename(mask_file).split(".png")[0] + ".JPG"
        if img_name in to_label_list:
            print(f"Uploading mask {upload_count} of {len(mask_files)}")
            mask = np.array(Image.open(mask_file))
            print(f"Performing upload of image: {img_name}")
            print(f"Image masks: {np.unique(mask)}")
            perform_upload(client, project, img_name, mask)
            upload_count += 1

if __name__ == "__main__":
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    project_id = config['project_id']
    api_key = config['api_key']
    mask_folder = config['mask_folder']

    client = labelbox.Client(api_key)
    project = client.get_project(project_id)
    labels = export_project_data(project)

    export_json = labels.result
    global_key_data_row_inputs = []
    to_label_list = []

    for img in export_json:
        # We assign our externel key (image name) as our global key too. You can also assign a custom global key.
        global_key_data_row_inputs.append({"data_row_id": img['data_row']['id'], "global_key": img['data_row']['external_id']})
    client.assign_global_keys_to_data_rows(global_key_data_row_inputs)
    # Load project data again with global keys
    labels = export_project_data(project)

    export_json = labels.result
    for img in export_json:
        workflow_status = img['projects'][project_id]['project_details']['workflow_status']
        # Specify which images you want to upload. You can delete this if you want to upload all.
        status = "TO_LABEL"
        if workflow_status == status: # Only upload images with status "TO_LABEL". Possible workflow statuses are: "TO_LABEL", "TO_REVIEW", "IN_REWORK", "DONE"
            to_label_list.append(img['data_row']['global_key'])
            to_label_list_set = set(to_label_list)
    print(f"{len(to_label_list)} images with status {status}")

    # Check for duplicates
    if len(to_label_list) != len(to_label_list_set):
        print("Duplicates found!")
    else:
        print("No duplicates found!")

    upload_masks(client, project, to_label_list, mask_folder, export_json) # Upload masks to data row with status "TO_LABEL"