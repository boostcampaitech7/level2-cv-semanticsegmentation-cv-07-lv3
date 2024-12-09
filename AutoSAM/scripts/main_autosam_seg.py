import argparse
import builtins
import math
import os
import gc
import yaml
import random
import wandb
import shutil
import time
import warnings
import pickle
import numpy as np
from datetime import datetime
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models


from loss_functions.dice_loss import DiceLoss

from models import sam_seg_model_registry
from dataset import generate_dataset, generate_test_loader


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-c', '--config', default='AutoSAM.yaml', type=str)
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_false',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--model_type', type=str, default="vit_l", help='path to splits file')
parser.add_argument('--src_dir', type=str, default=None, help='path to splits file')
parser.add_argument('--data_dir', type=str, default=None, help='path to datafolder')
parser.add_argument("--slice_threshold", type=float, default=0.05)
parser.add_argument("--num_classes", type=int, default=29)
parser.add_argument("--save_dir", type=str, default='autosam_crop')
parser.add_argument("--load_saved_model", action='store_true',
                        help='whether freeze encoder of the segmenter')
parser.add_argument("--saved_model_path", type=str, default='saved')
parser.add_argument("--dataset", type=str, default="xray")

def load_config(config_name):
    """Load config file from configs directory"""
    config_path = os.path.join('configs', config_name)
    if not os.path.exists(config_path):
        print(f'Config file not found: {config_path}')
        exit(1)
        
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f'Error loading config file: {e}')
            exit(1)
    return config

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.distributed = False
    args.multiprocessing_distributed = False

    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cfg = load_config(args.config)
    wandb.init(
        project = "Segmentation", 
        entity = 'jhs7027-naver', 
        group = cfg['WANDB']['GROUP'], 
        name = cfg['WANDB']['NAME'], 
        config = {
            "IMAGE_SIZE": cfg['DATASET'].get('IMAGE_SIZE'),
            "BATCH_SIZE": cfg['DATASET'].get('BATCH_SIZE'),
            "NUM_WORKERS": cfg['DATASET'].get('NUM_WORKERS'),
            "ENCODER": cfg['MODEL'].get('ENCODER'),
            "NUM_EPOCHS": cfg['TRAIN'].get('NUM_EPOCHS'),
            "VAL_EVERY": cfg['TRAIN'].get('VAL_EVERY'),
            "LEARNING_RATE": cfg['TRAIN'].get('LEARNING_RATE'),
            "WEIGHT_DECAY": cfg['TRAIN'].get('WEIGHT_DECAY'),
            "RANDOM_SEED": cfg['TRAIN'].get('RANDOM_SEED'),
            "LOSS_NAME": cfg['LOSS'].get('NAME'),
            "LOSS_WEIGHTS": cfg['LOSS'].get('WEIGHTS'),
            "OPTIMIZER_NAME": cfg['OPTIMIZER'].get('NAME'),
            "OPTIMIZER_LR": cfg['OPTIMIZER'].get('LR'),
            "OPTIMIZER_WEIGHT_DECAY": cfg['OPTIMIZER'].get('WEIGHT_DECAY'),
            "OPTIMIZER_BETAS": cfg['OPTIMIZER'].get('BETAS'),
            "OPTIMIZER_USE_TRITON": cfg['OPTIMIZER'].get('USE_TRITON'),
            "OPTIMIZER_MOMENTUM": cfg['OPTIMIZER'].get('MOMENTUM'),
            "OPTIMIZER_USE_LOOKAHEAD": cfg['OPTIMIZER'].get('USE_LOOKAHEAD'),
            "OPTIMIZER_LOOKAHEAD_K": cfg['OPTIMIZER'].get('LOOKAHEAD_K'),
            "OPTIMIZER_LOOKAHEAD_ALPHA": cfg['OPTIMIZER'].get('LOOKAHEAD_ALPHA'),
            "SCHEDULER_NAME": cfg['SCHEDULER'].get('NAME'),
            "SCHEDULER_STEP_SIZE": cfg['SCHEDULER'].get('STEP_SIZE'),
            "SCHEDULER_MILESTONES": cfg['SCHEDULER'].get('MILESTONES'),
            "SCHEDULER_GAMMA": cfg['SCHEDULER'].get('GAMMA'),
            "SCHEDULER_FACTOR": cfg['SCHEDULER'].get('FACTOR'),
            "SCHEDULER_PATIENCE": cfg['SCHEDULER'].get('PATIENCE'),
            "SCHEDULER_VERBOSE": cfg['SCHEDULER'].get('VERBOSE'),
            "SCHEDULER_T_MAX": cfg['SCHEDULER'].get('T_MAX'),
            "SCHEDULER_ETA_MIN": cfg['SCHEDULER'].get('ETA_MIN'),
            "VALIDATION_THRESHOLD": cfg['VALIDATION'].get('THRESHOLD'),
        }
    )
    args.gpu = gpu

    # remove unused memory
    gc.collect()
    torch.cuda.empty_cache()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.model_type=='vit_h':
        model_checkpoint = 'sam_vit_h_4b8939.pth'
    elif args.model_type == 'vit_l':
        model_checkpoint = 'sam_vit_l_0b3195.pth'
    elif args.model_type == 'vit_b':
        model_checkpoint = 'sam_vit_b_01ec64.pth'

    model = sam_seg_model_registry[args.model_type](num_classes=args.num_classes, checkpoint=model_checkpoint)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # freeze weights in the image_encoder
    for name, param in model.named_parameters():
        if param.requires_grad and "image_encoder" in name or "iou" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
        # param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(
                optimizer,
                mode=cfg['SCHEDULER'].get('MODE', 'min'),
                factor=cfg['SCHEDULER'].get('FACTOR', 0.1),
                patience=cfg['SCHEDULER'].get('PATIENCE', 10),
                verbose=cfg['SCHEDULER'].get('VERBOSE', True)
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler = generate_dataset(args, cfg)

    now = datetime.now()
    # args.save_dir = "output_experiment/Sam_h_seg_distributed_tr" + str(args.tr_size) # + str(now)[:-7]
    args.save_dir = "output_experiment/" + args.save_dir
    print(args.save_dir)
    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard' + str(gpu)))

    filename = os.path.join(args.save_dir, 'autosam_b')

    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_dice = -100

    for epoch in range(args.start_epoch, args.epochs):
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=epoch)

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, scheduler, epoch, args, writer, dice_loss, bce_loss)
        val_dice = validate(val_loader, model, epoch, args, writer, dice_loss)
        
        # wandb logging
        train_log_dict = {
            "train_epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        wandb.log(train_log_dict)

        # wandb validation logging
        val_log_dict = {
            "val_epoch": epoch + 1,
            "val_dice": 1 - val_dice,
            "val_loss": val_dice
        }
        wandb.log(val_log_dict)

        if 1 - val_dice > best_dice:
            best_dice = 1 - val_dice
            save_checkpoint(model, filename=filename, dice=np.round(1-val_dice, 4), epoch=epoch)
            print("saved ckpt at ", epoch)
            print("best dice:", best_dice)


    #test(model, args)


def train(train_loader, model, optimizer, scheduler, epoch, args, writer, dice_loss, bce_loss):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # switch to train mode
    model.train()

    end = time.time()
    for i, tup in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            img = tup[0].float().cuda(args.gpu, non_blocking=True)
            label = tup[1].cuda(args.gpu, non_blocking=True)
        else:
            img = tup[0].float()
            label = tup[1]
        b, c, h, w = img.shape

        # compute output
        # mask size: [batch*num_classes, num_multi_class, H, W], iou_pred: [batch*num_classes, 1]
        mask, iou_pred = model(img)
        mask = mask.view(b, -1, h, w)
        
        iou_pred = iou_pred.squeeze().view(b, -1)

        pred_softmax = torch.sigmoid(mask)
        loss = bce_loss(mask, label) + dice_loss(pred_softmax, label)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar('train_loss', loss, global_step=i + epoch * len(train_loader))

        if i % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss:.4f}'.format(epoch, i, len(train_loader), loss=loss.item()))

    scheduler.step(loss)
    
    return loss.item()


def validate(val_loader, model, epoch, args, writer, dice_loss):
    loss_list = []
    dice_list = []
    model.eval()

    with torch.no_grad():
        for i, tup in enumerate(val_loader):

            if args.gpu is not None:
                img = tup[0].float().cuda(args.gpu, non_blocking=True)
                label = tup[1].cuda(args.gpu, non_blocking=True)
            else:
                img = tup[0]
                label = tup[1]
            b, c, h, w = img.shape

            # compute output
            mask, iou_pred = model(img)
            mask = mask.view(b, -1, h, w)
            iou_pred = iou_pred.squeeze().view(b, -1)
            iou_pred = torch.mean(iou_pred)

            outputs = torch.sigmoid(mask)
            
            # Handle different output sizes
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = label.size(-2), label.size(-1)
            
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = dice_loss(outputs, label)  # self.ce_loss(pred, target.squeeze())
            loss_list.append(loss.item())

    print('Validating: Epoch: %2d Loss: %.4f IoU_pred: %.4f' % (epoch, np.mean(loss_list), iou_pred.item()))
    writer.add_scalar("val_loss", np.mean(loss_list), epoch)
    return np.mean(loss_list)


def save_checkpoint(model, filename, dice, epoch):
    # torch.save(state, filename)
    filename = filename + str(dice) + '_' + str(epoch) + '.pth'
    torch.save(model, filename)
    # shutil.copyfile(filename, 'model_best.pth.tar')


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

    # python main_moco.py --data_dir ./data/mmwhs/ --do_contrast --dist-url 'tcp://localhost:10001'
    # --multiprocessing-distributed --world-size 1 --rank 0
