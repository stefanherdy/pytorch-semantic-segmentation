#!/usr/bin/env python3

import os
import pickle
import json
from utils import *
import torch as t
import torch.nn as nn 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True

seed = 42 
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)

def main(args):
    args.save_dir = './checkpoints/'
    
    makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    dload_train, dload_valid = import_data(args, args.batch_size, args.project, args.resize, args.random_crop_size)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    f = get_model(device, args.num_classes)

    params = f.parameters() 
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.learnrate, betas=[.9, .999], weight_decay=0.0)
    else:
        optim = t.optim.SGD(params, lr=args.learnrate, momentum=.9, weight_decay=0.0)

    best_valid_acc = 0.0
    iteration = 0

    train_losses = []
    val_losses = []
    val_corr = []
    
    for epoch in range(args.epochs):
        iter_losses = []
        for i, (x_train, y_train) in tqdm(enumerate(dload_train)):
            x_train, y_train = next(iter(dload_train)) 
            x_train, y_train = x_train.to(device), y_train.to(device)

            Loss = 0.

            logits = f(x_train)
            l_dis = nn.CrossEntropyLoss()(logits, y_train)
            Loss += l_dis
            iter_losses.append(Loss.item())

            optim.zero_grad()
            Loss.backward()
            optim.step()

        if iteration % args.print_every == 0:
            acc = (logits.max(1)[1] == y_train).float().mean()
            print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                            iteration,
                                                                            l_dis.item(),
                                                                            acc.item()))

        iteration += 1

        train_losses.append(np.mean(iter_losses))

        if epoch % args.eval_every == 0:
            f.eval()
            with t.no_grad():
                correct, loss = eval_classification(f, dload_valid, device)
                val_losses.append(loss)
                val_corr.append(correct)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, "best_validation_ckpt.pt", args, device, dload_train, dload_valid)
            f.train()
        if epoch % args.ckpt_every == 0:
            checkpoint(f, f'ckeckpoint_{epoch}.pt', args, device, dload_train, dload_valid)
            
            # Losses are saved and can be loaded for further analysis
            # You can also plot them here using matplotlib
            with open("./records/trainlosses.txt" , "wb") as fp:
                pickle.dump(train_losses, fp)

            with open("./records/vallosses.txt" , "wb") as fp:
                pickle.dump(val_losses, fp)

            with open("./records/correct.txt" , "wb") as fp:
                pickle.dump(val_corr, fp)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pytorch Semantic Segmentation")
    parser.add_argument("--learnrate", type=int, default=0.0001, help='learn rate of optimizer')
    parser.add_argument("--optimizer", choices=['sgd', 'adam'], default='adam')
    parser.add_argument("--epochs", type=int, default=7000)
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=5, help="Epochs between print")
    parser.add_argument("--ckpt_every", type=int, default=2, help="Epochs between checkpoint save")
    parser.add_argument("--project", choices=['project_1', 'project_2', 'project_3'], default='project_1')
    parser.add_argument("--batch_size", type=int, default=2, help="Batch Size")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of classes of your training dataset")
    parser.add_argument("--resize", type=int, default=512, help="Size of images for resizing")
    parser.add_argument("--random_crop_size", type=int, default=256, help="Size of random crops. Must be smaller than resized images.")
    args = parser.parse_args()
    if args.random_crop_size > args.resize:
        raise Exception("Crop size (--random_crop_size) must be smaller than resized image (--resize)!")
    
    main(args)





    
    
