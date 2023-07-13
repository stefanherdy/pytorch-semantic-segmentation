
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
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
import random

im_sz = 256
n_ch = 3
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

    # datasets
    batch_sz = 8
    dload_train, dload_valid = import_data(args, batch_sz)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    num_classes = 8
    f = get_model(device, num_classes)
    

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
            max_it = len(dload_valid)


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
            fig = plt.figure()
            plt.plot(train_losses)
            plt.plot(val_losses)
            plt.title('Loss')
            plt.legend(('Training', 'Validation'))
            plt.grid()
            mintr = min(train_losses)
            minval = min(val_losses)
            mincorr = max(val_corr)
            with open("./records/trainlosses.txt" , "wb") as fp:   #Pickling
                pickle.dump(train_losses, fp)

            with open("./records/vallosses.txt" , "wb") as fp:   #Pickling
                pickle.dump(val_losses, fp)

            with open("./records/correct.txt" , "wb") as fp:   #Pickling
                pickle.dump(val_corr, fp)
            # test set
            #correct, loss = eval_classification(f, dload_test, device)
            #print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
            #checkpoint(f, replay_buffer, "last_ckpt.pt", args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    parser.add_argument("--learnrate", type=int, default=0.0001, help='learn rate of optimizer')
    parser.add_argument("--optimizer", choices=['sgd', 'adam'], default='adam')
    parser.add_argument("--epochs", type=int, default=7000)
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=5, help="Epochs between print")
    parser.add_argument("--ckpt_every", type=int, default=200, help="Epochs between checkpoint save")
    parser.add_argument("--energy", type=bool, default=True, help="Set p(x) optimization on(True)/off(False)")

    args = parser.parse_args()
    
    
    main(args)





    
    
