
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

im_sz = 256
n_ch = 3

seed= 42 

os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)

    

def get_model(device):
    
    unet = smp.Unet('resnet152', classes=8, activation=None, encoder_weights='imagenet')
    if t.cuda.is_available():
        unet.cuda()     

    
    
    print("Loading model")
    
    ckpt = t.load('./checkpoints/best_validation_ckpt.pt')
    unet.load_state_dict(ckpt["model_state_dict"])
    unet = unet.to(device)


    return unet

    
def predict(args, model, dload, device, set):
    iou_list = []
    correctlist = []
    for i, (x_p_d, y_p_d) in enumerate(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)

        model.eval()
        logits = model(x_p_d)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        correct = np.mean((logits.max(1)[1] == y_p_d).float().cpu().numpy())
        #print(np.unique(logits.max(1)[1].float().cpu().numpy()))
        #print(np.unique(y_p_d.float().cpu().numpy()))
        print('True: ' + str(i) + '_' + str(correct))
        correctlist.append(correct)
        logits_max = logits.max(1)[1].float().cpu().numpy()
        label = y_p_d.float().cpu().numpy() 
        IOU = mIOU(logits_max, label)
        iou_list.append(IOU)
        print(IOU)
        label = y_p_d.float().cpu().numpy()
        image = x_p_d.float().cpu().numpy()[0][:,:,:]
        image = np.moveaxis(image, 0, -1)

        logits_max = logits2rgb(logits_max[0,:,:])

        overlay = cv2.addWeighted(logits_max.astype(float)/255, 0.5, image.astype(float), 0.5, 0.0)

        label = logits2rgb(label[0,:,:])
        
        plot = False
        if plot == True:
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
            cv2.imwrite('./plots/plots_' + set + '/img' + str(i) + 'target.jpg', label)
            cv2.imwrite('./plots/plots_' + set + '/img' + str(i) + 'real.jpg', image*255)
            cv2.imwrite('./plots/plots_' + set + '/img' + str(i) + 'overlay.jpg', overlay_true*255)
        
    print('mIOU:')
    print(np.mean(iou_list))
    print('Accuracy:')
    print(np.mean(correctlist))
    


def evaluate(args):
    
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    set = 'project 3'
    batch_sz = 2

    
    dload_train, dload_valid = import_data(args, batch_sz, set)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    

        
    f = get_model(device, set)
    with t.no_grad():
        predict(args, f, dload_valid, device, set)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp_prob", type=int, default=0.05, help='probability for salt & pepper noise')
    parser.add_argument("--rn_std", type=int, default=0.05, help='standard devition for random noise')
    args = parser.parse_args()

    evaluate(args)





    
    
