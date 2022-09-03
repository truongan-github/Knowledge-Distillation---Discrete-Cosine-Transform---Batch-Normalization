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

import os
import argparse
import math
from vgg_ann_models_submit import *
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
parser.add_argument('--lr',                    default=0.1,       type=float, help='learning rate')
parser.add_argument('-b', '--batch_size',      default=128,       type=int,metavar='N', help='mini-batch size (default: 128)')
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
torch.cuda.manual_seed_all(seed)
num_train = 50000
num_test  = 10000
img_size  = 32
inp_maps  = 3
num_cls   = 10
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
model_str_use    = 'vgg16_cifar10_ann_lr.1_.1by100'+'_bs'+str(batch_size)+'_pixelexpanded_4avgpool'
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

# Data
print('==> Preparing data..')

#dataset             = 'tinyIMAGENET' # {'CIFAR10', 'CIFAR100', 'IMAGENET'}
dataset             = 'CIFAR10'
#usual

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

if dataset == 'CIFAR100':
    trainset  = CIFAR100(root='./data/cifar100', train=True, download=True,transform =transform_train)
    testset    = CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    
elif dataset == 'CIFAR10': 
    trainset   = CIFAR10(root='./data/cifar10', train=True, download=True,transform =transform_train)
    testset    = CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, generator=torch.Generator(device='cuda'))
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

# Model
print('==> Building model..')
model = VGG('VGG16', labels=labels)
model = model.cuda()
model = torch.nn.DataParallel(model).cuda()

use_cuda =torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
if(resume_from_ckpt):
   ckpt            = torch.load(ckpt_fname)
   start_epoch     = ckpt['start_epoch']
   end_epoch       = start_epoch+num_epochs
   accuracy_best   = ckpt['max_accuracy']
   epoch_best      = ckpt['epoch']
   model.load_state_dict(ckpt['model_state_dict'])
   optimizer.load_state_dict(ckpt['optim_state_dict'])
   print('Loaded ANN_VGG from {}\n'.format(ckpt_fname))
train_time = 0
print('********** ANN training and evaluation **********')
for epoch in range(start_epoch, end_epoch):
    loss_record = AverageMeter('Loss')
    acc_record   = AverageMeter('Acc@1')
    #Train model
    start_time = time.time()
    model.train()
    adjust_learning_rate(optimizer, epoch)
    for data, target in train_loader:
         # Load the inputs and targets
         if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()        
            
            if dataset=='CIFAR10' or dataset=='CIFAR100':
              data =rin(data)
            # Reset the gradient
            optimizer.zero_grad()
            
            # Perform forward pass and compute the target loss
            output = model(data)
            
            loss = F.cross_entropy(output,target)
            
            # Perform backward pass and update the weights
            loss.backward()
            optimizer.step()
            
            batch_acc = accuracy(output, target, topk=(1,))[0]
            loss_record.update(loss.item(), data.size(0))
            acc_record.update(batch_acc.item(), data.size(0))
        run_time = time.time() - start_time
        info = 'Train: epoch: {:03d}/{:03d}\t run_time:{:.2f}\t cls_loss:{:.4f}\t train_acc:{:.2f}\t'.format(
            epoch+1, epochs, run_time, loss_record.avg, acc_record.avg)
        print(info)
        
    # Evaluate classification accuracy on the test set
    model.eval()
        for data, target in test_loader:  
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            if dataset=='CIFAR10' or dataset=='CIFAR100':
                images =rin(images)
            with torch.no_grad():
                output = model(data)
                loss = F.cross_entropy(output,target)
            
            batch_acc = accuracy(output, target, topk=(1,))[0]
            loss_record.update(loss.item(), data.size(0))
            acc_record.update(batch_acc.item(), data.size(0))
            
        run_time = time.time() - start_time
        if epoch>30 and acc_record.avg<15.0:
            print('\n Quitting as the training is not progressing')
            exit(0)
        
        # Checkpoint SNN training and evaluation states
        if acc_record.avg >= old_max_accuracy:
            old_max_accuracy = acc_record.avg
            ckpt = {'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'start_epoch'     : epoch+1,
                    'accuracy_best'   : max_accuracy,
                    'epoch_best'      : epoch}
            torch.save(ckpt, name)
        if acc_record.avg > max_accuracy:
            max_accuracy = acc_record.avg    

        print('--Test: run_time: {:.2f}\t loss: {:.4f}\t test_acc: {:.2f}\t best: {:.2f}'.  format(
            run_time,
            loss_record.avg,
            acc_record.avg,
            max_accuracy
            )
        )
 print('Highest accuracy: {:.2f}'.format(max_accuracy))
