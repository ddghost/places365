# this code is modified from the pytorch example code: https://github.com/pytorch/examples/blob/master/imagenet/main.py
# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

import wideresnet
import pdb
import SENet
import progressbar
from PIL import Image
import pandas as pdb
import csv

from torchstat import stat

def main():
    model = SENet.se_resnet50(num_classes=365)
    stat(model, (3, 256, 256))
    model = SENet.se_resnet152(num_classes=365)
    stat(model, (3, 256, 256))

if __name__ == '__main__':
    main()

