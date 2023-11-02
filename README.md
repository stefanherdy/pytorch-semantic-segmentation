# pytorch-semantic-segmentation
Semantic Image Segmentation using Pytorch

This repository is about the full process from data preprocessing to the training to the evaluation of the models. 

![segmantation image](https://github.com/stefanherdy/pytorch-semantic-segmentation/blob/main/img/seg.png?raw=true)

Detailed descriptions of my code can be found on my articles on Medium:
- How to Augment Images for Semantic Segmentation
  https://medium.com/@stefan.herdy/how-to-augment-images-for-semantic-segmentation-2d7df97544de
- Pytorch Semantic Segmentation
  https://medium.com/@stefan.herdy/pytorch-semantic-image-segmentation-b726589662e3
- How to Evaluate Semantic Segmantation Models
  https://medium.com/@stefan.herdy/how-to-evaluate-semantic-segmantation-models-cd2539673701
- How to Download Labelbox Image Annotations
  https://medium.com/@stefan.herdy/how-to-export-labelbox-annotations-eedb8cb4f217
  


Installation

$ git clone https://github.com/stefanherdy/pytorch-semantic-segmentation.git

Usage

    - First, add your custom datasets to the input_data folder (You can download image annotations from Labelbox using download_mask.py).
    - If your Images have a background with label 0, you can run crop_bg.py to remove the background from the image.
    - Run train.py, optimize your hyperparameters
    - To evaluate the model rum evaluate.py


License

This project is licensed under the MIT License. ©️ 2023 Stefan Herdy
