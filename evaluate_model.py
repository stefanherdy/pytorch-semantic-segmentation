#!/usr/bin/env python3

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch as t
import torch.nn as nn 
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import cv2
import segmentation_models_pytorch as smp
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True

# For Reproducible results
seed= 42 
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)

def predict(args, model, dload, device, set):
    iou_list = []
    correctlist = []
    for i, (x_p_d, y_p_d) in enumerate(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)

        model.eval()
        logits = model(x_p_d)
        correct = np.mean((logits.max(1)[1] == y_p_d).float().cpu().numpy())
        correctlist.append(correct)
        logits_max = logits.max(1)[1].float().cpu().numpy()
        label = y_p_d.float().cpu().numpy() 

        IOU = mIOU(logits_max, label)
        iou_list.append(IOU)

        for j in range(label.shape[0]):
            label = y_p_d.float().cpu().numpy()
            image = x_p_d.float().cpu().numpy()[j][:,:,:]
            image = np.moveaxis(image, 0, -1)
            logits_max = logits.max(1)[1].float().cpu().numpy()
            logits_max = logits2rgb(logits_max[j,:,:])
            overlay = cv2.addWeighted(logits_max.astype(float)/255, 0.5, image.astype(float), 0.5, 0.0)
            label = logits2rgb(label[j,:,:])
            
            
            if args.plot_results == True:
                fig, axs = plt.subplots(1,3)
                axs[0].imshow(label)
                axs[1].imshow(image)
                axs[2].imshow(overlay)
                plt.show()
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                overlay_true = cv2.cvtColor(overlay.astype('float32'), cv2.COLOR_RGB2BGR)
                label = cv2.cvtColor(label.astype('float32'), cv2.COLOR_RGB2BGR)
                makedirs(f'./plots/plots_{set}/')
                cv2.imwrite(f'./plots/plots_{set}/img{i}target.jpg', label)
                cv2.imwrite(f'./plots/plots_{set}/img{i}real.jpg', image*255)
                cv2.imwrite(f'./plots/plots_{set}/img{i}overlay.jpg', overlay_true*255)
        
    print('mIOU:')
    print(np.mean(iou_list))
    print('Accuracy:')
    print(np.mean(correctlist))
    


def evaluate(args):
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    
    dload_train, dload_valid = import_data(args, args.batch_size, args.project, args.resize, args.random_crop_size)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')   
    model = get_model(device, args.num_classes)
    ckpt = t.load('./checkpoints/best_validation_ckpt.pt') 
    model.load_state_dict(ckpt["model_state_dict"])
    with t.no_grad():
        predict(args, model, dload_valid, device, args.project)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Semantic Segmantation Model Evaluation')
    parser.add_argument("--resize", type=int, default=512, help="Size of images for resizing")
    parser.add_argument("--random_crop_size", type=int, default=256, help="Size of random crops. Must be smaller than resized images.")
    parser.add_argument("--project", choices=['project_1', 'project_2', 'project_3'], default='project_1')
    parser.add_argument("--batch_size", type=int, default=2, help="Batch Size")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of classes of your training dataset")
    parser.add_argument("--plot_results", type=bool, default=False, help="Set to True if you want to plot your results.")
    args = parser.parse_args()

    evaluate(args)





    
    
