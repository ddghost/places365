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

parser = argparse.ArgumentParser(description='PyTorch Places365')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

device_ids = [0,1,2,3]
model  = SENet.se_resnet(365)

def checkConvParameter(model):
	for m in model.modules():
		if isinstance(m, nn.Conv2d) and m.kernel_size == (3,3):
			kernelsNum, inplanes, w, h = m.weight.shape
			
			weight = m.weight.view(kernelsNum, inplanes*w*h).abs()
			print( abs(weight.sum(1)) / abs(weight.sum(1).max() )  )
			
	print('!!!!!!!!!!!!!!!!!!!!')
	for m in model.modules():
		if isinstance(m, nn.Conv2d) and m.kernel_size == (3,3):
			kernelsNum, inplanes, w, h = m.weight.shape
			m.weight.permute(1,0,2,3)
			weight = m.weight.view(inplanes, kernelsNum*w*h).abs()
			print( abs(weight.sum(1)) / abs(weight.sum(1).max() )  )
		
			
def main():
	args = parser.parse_args()
	
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
			del checkpoint
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))
	else:
		print(model)
	checkConvParameter(model)
	
if __name__ == '__main__':
    main()