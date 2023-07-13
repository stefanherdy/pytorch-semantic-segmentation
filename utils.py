import os
import pickle
import json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data import DataLoader
import pathlib
from customdatasets import CustomDataSet
from transformations import Compose, DenseTarget, RandomFlip, Resize_Sample
from transformations import MoveAxis, Normalize01, RandomCrop
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from os import walk
import torch as t
import numpy as np
import torch.nn as nn


def get_files(path):
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        for names in filenames:
            files.append(dirpath + '/' + names)
    return files

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_model(device, cl):
    
    unet = smp.Unet('resnet152', classes=cl, activation=None, encoder_weights='imagenet')

    if t.cuda.is_available():
        unet.cuda()         
    
    unet = unet.to(device)
    return unet


def import_data(args, batch_sz, set = 'project_3'):

    root = pathlib.Path('./')
    if set == 'project_1':
        inputs = get_files('./input_data/project_1/image/')
        targets = get_files('./input_data/project_1/target/')

    if set == 'project_2':
        inputs = get_files('./input_data/project_2/image/')
        targets = get_files('./input_data/project_2/target/')

    if set == 'project_3':
        inputs = get_files('./input_data/project_3/image/')
        targets = get_files('./input_data/project_3/target/')

    split = 0.8  

    inputs_train, inputs_valid = train_test_split(
        inputs,
        random_state=42,
        train_size=split,
        shuffle=True)

    targets_train, targets_valid = train_test_split(
        targets,
        random_state=42,
        train_size=split,
        shuffle=True)


    
    transforms = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01(),
        RandomCrop(),
        RandomFlip()
        ])
    

    # train dataset
    dataset_train = CustomDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        transform=transforms)


    # validation dataset
    dataset_valid = CustomDataSet(inputs=inputs_valid,
                                        targets=targets_valid,
                                        transform=transforms)


    batchsize = batch_sz


    # train dataloader
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=batchsize,
                                    shuffle=True
                                    )

    # validation dataloader
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=batchsize,
                                    shuffle=True)

    
    
    return dataloader_training, dataloader_validation



def eval_classification(f, dload, device):
    corrects, losses = [], []
    for input, target in dload:
        input, target = input.to(device), target.to(device)
        logits = f(input)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, target).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == target).float().cpu().numpy()
        logits_max = logits.max(1)[1].float().cpu().numpy()
        label = target.float().cpu().numpy()
        
        # You can print the evaluation results, if needed
        '''
        fig, axs = plt.subplots(2)
        axs[0].imshow(logits_max[0,:,:])
        axs[1].imshow(label[0,:,:])
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


def checkpoint(f, tag, args, device, dload_train, dload_valid):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "train": dload_train,
        "valid": dload_valid
        #"sample": dload_sample,
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


def logits2rgb(img):
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


def mIOU(pred, label, num_classes=8):
    
    iou_list = list()
    present_iou_list = list()

    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.sum().item() == 0:
            iou_now = float('nan')
        else:
            #inters =np.logical_and(pred_inds, target_inds).sum().item()
            intersection_now = (pred_inds[target_inds]).sum().item()
            union_now = pred_inds.sum().item() + target_inds.sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
        miou = np.mean(present_iou_list)
    return miou