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
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='seresnet50_new',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: seresnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=70, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=365, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='places365',help='which dataset to train')
device_ids = [2,3]
ini_device = 2
best_prec1 = 0


class optimizerController(object):
    def __init__(self, net, trainEpoches, iniLr=1e-1, finalLr=1e-4):
        self.trainEpoches = trainEpoches
        if(hasattr(net, 'module') ):
            model = net.module
        else:
            model = net
        
        self.modelOptimiazer = torch.optim.SGD(model.parameters(), lr=finalLr, momentum=0.9, weight_decay=1e-4)
        self.optimizer0 = torch.optim.SGD(model.getParameters(0), lr=iniLr, momentum=0.9, weight_decay=1e-4) 
        self.optimizer1 = torch.optim.SGD(model.getParameters(1), lr=iniLr, momentum=0.9, weight_decay=1e-4) 
        self.optimizer2 = torch.optim.SGD(model.getParameters(2), lr=iniLr, momentum=0.9, weight_decay=1e-4) 
        self.optimizer3 = torch.optim.SGD(model.getParameters(3), lr=iniLr, momentum=0.9, weight_decay=1e-4) 
        self.optimizer4 = torch.optim.SGD([{'params':  model.getParameters(4)},
                                    {'params':  model.getParameters(5)} ], 
                                    lr=iniLr, momentum=0.9, weight_decay=1e-4) 
        
        self.scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer0, T_max=trainEpoches[0],eta_min=finalLr)
        self.scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer1, T_max=trainEpoches[1],eta_min=finalLr)
        self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer2, T_max=trainEpoches[2],eta_min=finalLr)
        self.scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer3, T_max=trainEpoches[3],eta_min=finalLr)
        self.scheduler4 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer4, T_max=trainEpoches[4],eta_min=finalLr)
        
    def optimizerStep(self, epoch):
        if(epoch < self.trainEpoches[0] ):
            self.optimizer0.step()
        if(epoch < self.trainEpoches[1]):
            self.optimizer1.step()
        if(epoch < self.trainEpoches[2]):
            self.optimizer2.step()
        if(epoch < self.trainEpoches[3]):
            self.optimizer3.step()
            
        if(epoch < self.trainEpoches[4]):
            self.optimizer4.step()
        else:
            self.modelOptimiazer.step()
            
    def optimizerZero_grad(self, epoch):
        if(epoch < self.trainEpoches[0] ):
            self.optimizer0.zero_grad()
        if(epoch < self.trainEpoches[1]):
            self.optimizer1.zero_grad()
        if(epoch < self.trainEpoches[2]):
            self.optimizer2.zero_grad()
        if(epoch < self.trainEpoches[3]):
            self.optimizer3.zero_grad()
            
        if(epoch < self.trainEpoches[4]):
            self.optimizer4.zero_grad()
        else:
            self.modelOptimiazer.zero_grad()        
     
    def schedulerStep(self):
        self.scheduler0.step()
        self.scheduler1.step()
        self.scheduler2.step()
        self.scheduler3.step()
        self.scheduler4.step()



def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    torch.cuda.set_device(ini_device)
    # create model
    print("=> creating model '{}'".format(args.arch))

    model  = SENet.se_resnet50(num_classes=args.num_classes)
    model = torch.nn.DataParallel(model, device_ids).cuda()
    # optionally resume from a checkpoint
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
            return
    else:
        print(model)
    
    
    
    
    cudnn.benchmark = True

    train_loader, val_loader = getDataLoader(args.data)
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    
    if args.evaluate:
        print(validate(val_loader, model, criterion))
        return
        
        
    #trainEpoch = [4,8,16,32,64]
    trainEpoch = [40,40,40,40,40]
    opController = optimizerController(model, trainEpoch, iniLr=1e-1, finalLr=1e-4)
    for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            
            
        if(trainEpoch[0] <= epoch and epoch < trainEpoch[1]):
            model.module.frezzeFromShallowToDeep(0)
        elif(trainEpoch[1] <= epoch and epoch < trainEpoch[2]):
            model.module.frezzeFromShallowToDeep(1)
        elif(trainEpoch[2] <= epoch and epoch < trainEpoch[3]):
            model.module.frezzeFromShallowToDeep(2)
        elif(trainEpoch[3] <= epoch and epoch < trainEpoch[4]):
            model.module.frezzeFromShallowToDeep(3)
        else:
            model.module.frezzeFromShallowToDeep(-1)
            
        train(train_loader, model, criterion, opController, epoch)
        opController.schedulerStep()
            # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, args.arch.lower())
       



def getDataLoader(dataDir):
    # Data loading code
    traindir = os.path.join(dataDir, 'train')
    valdir = os.path.join(dataDir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


def train(train_loader, model, criterion, opController, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    bar = progressbar.progressbar(len(train_loader))
    end = time.time()
    start = time.perf_counter()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        opController.optimizerZero_grad(epoch)
        loss.backward()
        opController.optimizerStep(epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #bar.clear()
        if i % args.print_freq == 0:
            print('\rEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        #bar.output(i+1)
    #print()
    print('Epoch waste time {}s'.format(time.perf_counter()- start) )

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            '''
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
            '''
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0) 
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

