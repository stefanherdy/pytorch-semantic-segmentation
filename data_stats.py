#!/usr/bin/env python3

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

set = ('project_3')
rootdir = './input_data/' + set + '/mask/'

classes = ['class_1', "class_2", 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9']
# From Labelbox stats (can also be computed)
classnum = [32, 41, 25, 47, 42, 58, 43, 39]

classdist = np.zeros(8)
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        img = cv2.imread(rootdir + file)
        for i in range(len(classes)):
            class_inds = (img == i)
            all = class_inds.sum().item()
            classes = np.zeros(8)
            classes[i] = all
            classdist = classdist + classes

classdist = classdist/(np.sum(classdist))*100
print(classdist)

# Customize colors
red = [200, 0, 10]
green = [187,207, 74]
blue = [0,108,132]
yellow = [255,204,184]
black = [0,0,0]
white = [226,232,228]
cyan = [174,214,220]
orange = [232,167,53]

classes = [classdist for _,classdist in sorted(zip(classdist,classes))]
classdist = np.sort(classdist)
classnum = np.sort(classnum)

colours = np.array([red, green, blue, yellow, black, white, cyan, orange])/255
colours = np.flip(colours, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(17, 7)
ax1.bar(classes,classnum, width = 0.4, color=colours)
cmap = dict(zip(classnum, colours))
patches = [Patch(color=v, label=k) for k, v in cmap.items()]

ax1.set_xlabel("Classes", fontsize=14)
ax1.set_ylabel("Number of Images", fontsize=14)
ax1.set_title('Total Class Distribution', fontsize=15)
ax1.text(0.05, 0.95, "a)", fontweight="bold", transform=ax1.transAxes)
ax2.set_xlabel("Classes", fontsize=14)
ax2.set_ylabel("Percentage of total Classifications", fontsize=14)
ax2.set_title('Overall Pixel Distribution', fontsize=15)
ax2.text(0.05, 0.95, "b)", fontweight="bold", transform=ax2.transAxes)
ax1.set_xticklabels([])
ax2.set_xticklabels([])
fig.legend(title='', labels=classes, handles=patches, loc='lower center', borderaxespad=-0.5, borderpad=1, fontsize=30, frameon=False, ncol=8, prop={'size': 14}) #, 'style': 'italic'})
fig.suptitle("Data Class Distribution", fontsize=17)
plt.show()