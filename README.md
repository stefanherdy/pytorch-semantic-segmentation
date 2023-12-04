# pytorch-semantic-segmentation
Semantic Image Segmentation using Pytorch

This repository is about the full process from data preprocessing to the training to the evaluation of the models. 

![segmantation image](https://github.com/stefanherdy/pytorch-semantic-segmentation/blob/main/img/seg.png?raw=true)

Detailed descriptions of my code can be found on my articles on Medium:
- How to Augment Images for Semantic Segmentation
  
  https://medium.com/@stefan.herdy/how-to-augment-images-for-semantic-segmentation-2d7df97544de?source=friends_link&sk=3978b373c7192775e7acaed4ce357ad9
- Pytorch Semantic Segmentation
  
  https://medium.com/@stefan.herdy/pytorch-semantic-image-segmentation-b726589662e3?source=friends_link&sk=296c6d5a19044cef272270c7bd39f814
- How to Evaluate Semantic Segmantation Models
  
  https://medium.com/@stefan.herdy/how-to-evaluate-semantic-segmantation-models-cd2539673701?source=friends_link&sk=94b359ecbb04c3413b3a6d82d42ea21c
- How to Download Labelbox Image Annotations
  
  https://medium.com/@stefan.herdy/how-to-export-labelbox-annotations-eedb8cb4f217?source=friends_link&sk=fb57499999b624245c4d4713f977a73c
  
  File: download_mask.py
- How to Upload Labelbox Image Annotations to Perform Active Learning
  
  https://medium.com/@stefan.herdy/active-learning-with-labelbox-how-to-upload-your-label-masks-99feef86017b?source=friends_link&sk=6525eb8fd4ebfe9b317ddc17247bcd2a
  
  File: upload_mask.py

To perform further analysis (Curvature Computation) of the segmentation masks you can read the following article:
- Compute the Curvature of a Binary Mask in Python
  
  https://medium.com/@stefan.herdy/compute-the-curvature-of-a-binary-mask-in-python-5087a88c6288?source=friends_link&sk=7e1bf51044644b678d75e82d310357af
  


Installation

$ git clone https://github.com/stefanherdy/pytorch-semantic-segmentation.git

Usage

    - To train the semantic segmentation model run train.py with "python train.py".
        You can specify the following parameters:
        --batch_size, type=int, default=8, help="Batch Size"
        --learnrate, type=int, default=0.0001, help='learn rate of optimizer'
        --optimizer, choices=['sgd', 'adam'], default='adam'
        --eval_every, type=int, default=1, help="Epochs between evaluation"
        --print_every, type=int, default=1, help="Epochs between print"
        --ckpt_every, type=int, default=20, help="Epochs between checkpoint save"
        --num_classes, type=int, default=8, help="Number of classes"
        --project, choices=['project_1', 'project_2', 'project_3'], default='project_1'
        --resize, type=int, default=512, help="Size of images for resizing"
        --random_crop_size, type=int, default=256, help="Size of random crops. Must be smaller than resized images."

        Example usage:
        "python train.py --eval_every 10 --learnrate 0.00001 --batch_size 16

        
    - To evaluate the model run evaluate_model.py with "python evaluate_model.py".
        You can specify the following parameters:
        --num_classes, type=int, default=8, help="Number of classes"
        --batch_size, type=int, default=8, help="Batch Size"
        --project, choices=['project_1', 'project_2', 'project_3'], default='project_1'
        --resize, type=int, default=512, help="Size of images for resizing"
        --random_crop_size, type=int, default=256, help="Size of random crops. Must be smaller than resized images."
        --plot_results, type=bool, default=False, help="Set to True if you want to plot your results. Set False to save results as images"

        Example usage:
        "python evaluate_model.py --batch_size 16

        Make sure you performed the training before, so that the models can be loaded for evaluation.
        Parameters like num_classes, random_crop_size and resize should match your training settings.


License

This project is licensed under the MIT License. ©️ 2023 Stefan Herdy
