import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn
from   tensorboard_logger import configure, log_value
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import CIFAR100,CIFAR10

import os
import argparse
import math
import sys
from vgg_ann_models_submit import *
# from utils import progress_bar
import time

def rin(input,b=4,s=2):
  x=int(((input.shape[2]-b)/s)+1)*b
  y=int(((input.shape[3]-b)/s)+1)*b
  output = torch.zeros(input.shape[0],input.shape[1],x,y)
  m=-1
  
  for i in range(0, input.shape[2] - b + 1, s):
    m=m+1
    n=-1
    for j in range(0, input.shape[3] - b + 1, s):
      n=n+1
      output[:,:,m*b : (m+1)*b,n*b : (n+1)*b]=input[:, :, i:i+b, j:j+b]
      
  return output

class DCT2(nn.Module):
    def __init__(self, block_size=4, p=0, mode = 'random', mean = None, std=None, device = 'cpu'):

      super(DCT2, self).__init__()
      ### forming the cosine transform matrix
      self.block_size = block_size
      self.device = device
      self.mean =mean
      self.std =std
      self.Q = torch.zeros((self.block_size,self.block_size)).to(self.device)
      
      self.Q[0] = math.sqrt( 1.0/float(self.block_size) )
      for i in range (1,self.block_size,1):
        for j in range(self.block_size):
          self.Q[i,j] = math.sqrt( 2.0/float(self.block_size) ) * math.cos( float((2*j+1)*math.pi*i) /float(2.0*self.block_size) )

      

    def rgb_to_ycbcr(self,input):
        
        # input is mini-batch N x 3 x H x W of an RGB image
        #output = Variable(input.data.new(*input.size())).to(self.device)
        output = torch.zeros_like(input).to(self.device)
        input = (input * 255.0)
        output[:, 0, :, :] = input[:, 0, :, :] * 0.299+ input[:, 1, :, :] * 0.587 + input[:, 2, :, :] * 0.114 
        output[:, 1, :, :] = input[:, 0, :, :] * -0.168736 - input[:, 1, :, :] *0.331264+ input[:, 2, :, :] * 0.5 + 128
        output[:, 2, :, :] = input[:, 0, :, :] * 0.5 - input[:, 1, :, :] * 0.418688- input[:, 2, :, :] * 0.081312+ 128
        return output/255.0

    def ycbcr_to_freq(self,input): 
 
        
        output = torch.zeros_like(input).to(self.device)
        a=int(input.shape[2]/self.block_size)
        b=int(input.shape[3]/self.block_size)
       
        # Compute DCT in block_size x block_size blocks 
        for i in range(a):
            for j in range(b):
                output[:,:,i*self.block_size : (i+1)*self.block_size,j*self.block_size : (j+1)*self.block_size] = torch.matmul(torch.matmul(self.Q, input[:, :, i*self.block_size : (i+1)*self.block_size, j*self.block_size : (j+1)*self.block_size]), self.Q.permute(1,0).contiguous() )
               
        return output 

    def forward(self, x):
        #return self.ycbcr_to_freq( self.rgb_to_ycbcr(x) )
        if (x.shape[1]==3):
          return self.ycbcr_to_freq( self.rgb_to_ycbcr(x) )
        else:
          return self.ycbcr_to_freq(x )  

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.001 ** (epoch // 100))
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
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
parser = argparse.ArgumentParser(description='PyTorch tinyimagenet Training')
parser.add_argument('--lr',                    default=0.001,       type=float, help='learning rate')
parser.add_argument('-b', '--batch_size',      default=16,       type=int,metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--resume', '-r',          action='store_true',           help='resume from checkpoint')
parser.add_argument('--seed',                  default=0,         type=int,   help='Random seed')
parser.add_argument('--ckpt_dir',              default=None,      type=str,   help='Checkpoint dir. If set to none, default dir is used')
parser.add_argument('--ckpt_intrvl',           default=1,         type=int,   help='Number of epochs between successive checkpoints')
parser.add_argument('--num_epochs',            default=312,       type=int,   help='Number of epochs for backpropagation')
parser.add_argument('--resume_from_ckpt',      default=0,         type=int,   help='Resume from checkpoint?')
parser.add_argument('--tensorboard',           default=0,         type=int,   help='Log progress to TensorBoard')
global args
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
# Initialize seed
#--------------------------------------------------
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_train = 50000
num_test  = 10000
img_size  = 32
inp_maps  = 3
num_cls   = 10
kd_T      = 4.0
ce_weight = 0.1
kd_weight = 0.9
test_error_best = 100 
start_epoch     = 0
num_epochs      = args.num_epochs
end_epoch       = start_epoch+num_epochs
batch_size      = args.batch_size
ckpt_dir         = args.ckpt_dir
ckpt_intrvl      = args.ckpt_intrvl
resume_from_ckpt = True if args.resume_from_ckpt else False
#model_str_use    = 'vgg11_cifar100_ann'+'_bs'+str(batch_size)+'_new_'+str(args.lr)+'lrby5_every30epoch'
# model_str_use    = 'vgg9_cifar10_ann_lr.1_.1by100'+'_bs'+str(batch_size)+'_pixelexpanded_4avgpool'
# model_str_use    = 'vgg11_cifar10_ann_lr.1_.1by100'+'_bs'+str(batch_size)+'_pixelexpanded_4avgpool'
model_str_use    = 'vgg16_cifar10_base_ann_lr.1_.1by100'+'_bs'+str(batch_size)+'_pixelexpanded_4avgpool'
#model_str_use    = 'vgg13_tinyimgnet_4*4dctbnmaxpool_ann_lr.01_.1by100'+'_bs'+str(batch_size)+'_wd1e-4'

if(ckpt_dir is None):
   ckpt_dir = './vgg16_snn_surrgrad_backprop/CHECKPOINTS/'+model_str_use
   ckpt_dir = os.path.expanduser(ckpt_dir)
   if(ckpt_intrvl > 0):
      if(not os.path.exists(ckpt_dir)):
         os.mkdir(ckpt_dir)
ckpt_fname  = ckpt_dir+'/ckpt.pth'
# Use TensorBoard?
tensorboard = True if args.tensorboard else False
gpu = True
use_cuda = torch.cuda.is_available()

if torch.cuda.is_available() and gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
# Data
print('==> Preparing data..')

#dataset             = 'tinyIMAGENET' # {'CIFAR10', 'CIFAR100', 'IMAGENET'}
dataset             = 'CIFAR10'
#usual

# usual imgnet stat from repos
#normalize       = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# calculated itiny-mgnet stat 
#normalize       = transforms.Normalize(mean = [0.48, 0.448, 0.3975], std = [0.277, 0.269, 0.282])

if dataset == 'CIFAR100':
    normalize   = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    labels      = 100 
elif dataset == 'CIFAR10':
    normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    labels      = 10
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
transform_test = transforms.Compose([transforms.ToTensor(), normalize])

# if dataset == 'CIFAR10':
#     trainset    = datasets.CIFAR10(root = './cifar_data', train = True, download = True, transform=transform_train)
#     testset     = datasets.CIFAR10(root='./cifar_data', train=False, download=True, transform=transform_test)
#     labels      = 10

# elif dataset == 'CIFAR100':
#     trainset    = datasets.CIFAR100(root = './cifar_data', train = True, download = True, transform=transform_train)
#     testset     = datasets.CIFAR100(root='./cifar_data', train=False, download=True, transform=transform_test)
#     labels      = 100

# elif dataset == 'IMAGENET':
#     labels      = 1000
#     traindir    = os.path.join('/local/scratch/a/imagenet/imagenet2012/', 'train')
#     valdir      = os.path.join('/local/scratch/a/imagenet/imagenet2012/', 'val')
#     trainset    = datasets.ImageFolder(
#                         traindir,
#                         transforms.Compose([
#                             transforms.RandomResizedCrop(224),
#                             transforms.RandomHorizontalFlip(),
#                             transforms.ToTensor(),
#                             normalize,
#                         ]))
#     testset     = datasets.ImageFolder(
#                         valdir,
#                         transforms.Compose([
#                             transforms.Resize(256),
#                             transforms.CenterCrop(224),
#                             transforms.ToTensor(),
#                             normalize,
#                         ])) 
# elif dataset == 'tinyIMAGENET':
#     labels      = 200
#     # adding the tinyimagenet directory
#     traindir    = os.path.join('/home/nano01/a/banerj11/srinivg_BackProp_CIFAR10/sayeed/tiny-imagenet-200/', 'train')
#     valdir      = os.path.join('/home/nano01/a/banerj11/srinivg_BackProp_CIFAR10/sayeed/tiny-imagenet-200/', 'val')
   
# #    traindir    = os.path.join('/local/scratch/a/chowdh23/data/tiny-imagenet-200/', 'train')
# #    valdir      = os.path.join('/local/scratch/a/chowdh23/data/tiny-imagenet-200/', 'val')
#     trainset    = datasets.ImageFolder(
#                         traindir,
#                         transforms.Compose([
#                             transforms.RandomResizedCrop(64),
#                             transforms.RandomHorizontalFlip(),
#                             transforms.RandomVerticalFlip(),
#                             transforms.ToTensor(),
#                             normalize,
#                         ]))
#     testset     = datasets.ImageFolder(
#                         valdir,
#                         transforms.Compose([
#                             #transforms.Resize(256),
#                             #transforms.CenterCrop(224),
#                             transforms.ToTensor(),
#                             normalize,
#                         ]))

if dataset == 'CIFAR100':
    trainset  = CIFAR100(root='./data/cifar100', train=True, download=True,transform =transform_train)
    testset    = CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    
elif dataset == 'CIFAR10': 
    trainset   = CIFAR10(root='./data/cifar10', train=True, download=True,transform =transform_train)
    testset    = CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, generator=torch.Generator(device='cuda'))
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

ckpt_path = "./vgg16_snn_surrgrad_backprop/CHECKPOINTS/vgg16_cifar10_ann_lr.1_.1by100_bs128_pixelexpanded_4avgpool/ckpt.pth"
state_dict = torch.load(ckpt_path)['model_state_dict']
l_model = VGG('VGG16', labels=labels)
l_model = torch.nn.DataParallel(l_model).cuda()
l_model.load_state_dict(state_dict)
if use_cuda and gpu:
    l_model.cuda()

l_model.eval()
acc_record = AverageMeter('Acc@1')
loss_record = AverageMeter("Loss")
for data, target in testloader:
    if torch.cuda.is_available() and gpu:
        data, target = data.cuda(), target.cuda() 
    with torch.no_grad():
        if dataset=='CIFAR10' or dataset=='CIFAR100':
            data=rin(data)
        output = l_model(data)
        loss = F.cross_entropy(output, target)

    batch_acc = accuracy(output, target, topk=(1,))[0]
    loss_record.update(loss.item(), data.size(0))
    acc_record.update(batch_acc.item(), data.size(0))

info = 'large ann: \tAccuracy: {:.4f}\tLoss: {:.4f}'.format(acc_record.avg,loss_record.avg)
print(info)

# Model
print('==> Building model..')
model = VGG('VGG9', labels=labels)
if torch.cuda.is_available() and gpu:
        model = model.cuda()

model = torch.nn.DataParallel(model).cuda()
print(model)

# device = torch.device("cuda" if use_cuda else "cpu")
# m=DCT2(block_size=4, device = device).to(device)

# criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
print(' {}'.format(optimizer))

#optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4, amsgrad=False)
if(resume_from_ckpt):
   ckpt            = torch.load(ckpt_fname)
   start_epoch     = ckpt['start_epoch']
   end_epoch       = start_epoch+num_epochs
   test_error_best = ckpt['test_error_best']
   epoch_best      = ckpt['epoch_best']
   train_time      = ckpt['train_time']
   model.load_state_dict(ckpt['model_state_dict'])
   optimizer.load_state_dict(ckpt['optim_state_dict'])
   print('Loaded ANN_VGG from {}\n'.format(ckpt_fname))
train_time = 0
print('********** ANN training and evaluation **********')
max_accuracy = 0
old_max_accuracy = 0
if use_cuda and True:
        model.cuda()
for epoch in range(start_epoch, end_epoch):
    start_time = time.time()
    #Train
    model.train()
    loss_ce_record = AverageMeter('LossCE')
    loss_kd_record = AverageMeter('LossKD')
    acc_record = AverageMeter('Acc@1')
    adjust_learning_rate(optimizer, epoch)
    for inputs, targets in trainloader:
        if use_cuda and gpu:
            data, target = data.cuda(), target.cuda() 
        # Load the inputs and targets        
        if dataset=='CIFAR10' or dataset=='CIFAR100':
            inputs =rin(inputs)
        # print(inputs)
        optimizer.zero_grad()
        
        output = model(inputs)
        # print(output)
        log_nor_output = F.log_softmax(output / kd_T, dim=1)
        with torch.no_grad():
          knowledge = l_model(inputs)
          nor_knowledge = F.softmax(knowledge / kd_T, dim=1)

        loss_ce   = F.cross_entropy(output, targets)
        # print(loss_ce)
        loss_kd = F.kl_div(log_nor_output, nor_knowledge, reduction='batchmean') * kd_T * kd_T
        loss = ce_weight * loss_ce + kd_weight * loss_kd

        # Perform backward pass and update the weights
        loss.backward()
        optimizer.step()
        
        batch_acc = accuracy(output, targets, topk=(1,))[0]
        loss_ce_record.update(loss_ce.item(), inputs.size(0))
        loss_kd_record.update(loss_kd.item(), inputs.size(0))
        acc_record.update(batch_acc.item(), inputs.size(0))
        
        # print(loss_ce_record)
    run_time = time.time() - start_time
    info = 'Train: Epoch:{:03d}/{:03d}\t Time:{:.3f}\t Cross-entropy loss:{:.3f}\t Kulback-Leibner Loss:{:.3f}\t Accuracy:{:.2f}'.format(
            epoch+1, num_epochs, run_time, loss_ce_record.avg, loss_kd_record.avg, acc_record.avg)
    print(info)

    # Evaluate classification accuracy on the test set
    model.eval()
    start_time = time.time()
    loss_record = AverageMeter('Loss')
    acc_record = AverageMeter('Acc@1')
    for images, labels in testloader:
      if use_cuda and gpu:
        images, labels = images.cuda(), labels.cuda()
      if dataset=='CIFAR10' or dataset=='CIFAR100':
        images =rin(images)
      with torch.no_grad():                      
        out     = model(images)
        loss   = F.cross_entropy(out, labels)
             
      batch_acc = accuracy(out, labels, topk=(1,))[0]
      loss_record.update(loss.item(), images.size(0))
      acc_record.update(batch_acc.item(), images.size(0))
          
    run_time = time.time() - start_time
    if acc_record.avg >= old_max_accuracy:
            old_max_accuracy = acc_record.avg
            # Checkpoint SNN training and evaluation states
            ckpt = {'model_state_dict': model.state_dict(),
                      'optim_state_dict': optimizer.state_dict(),
                      'start_epoch'     : epoch+1,
                      'test_error_best' : loss_record.avg,
                      'epoch_best'      : epoch,
                      'train_time'      : train_time}
            torch.save(ckpt, ckpt_fname)
    if acc_record.avg > max_accuracy:
            max_accuracy = acc_record.avg
    info = '--Test: \t Time:{:.2f}\t Test loss:{:.4f} \t Accuracy:{:.2f}\t Best accuracy:{:.2f}'.format(
                run_time, loss_record.avg, acc_record.avg,max_accuracy)
    print(info)        
print('Highest base ann accuracy: {:.4f}'.format(max_accuracy))
