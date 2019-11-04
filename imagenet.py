import argparse
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tensorboardX import SummaryWriter

import models.backbones as customized_models
import utils
from utils.distributed_utils import dist_init, average_gradients, DistModule

default_model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(
    name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(
            customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('--use_sw', action='store_true',
                    help='use switchable whitening or not')
parser.add_argument('--sw_type', default=2, type=int,
                    help='switchable whitening type')
parser.add_argument('--num_pergroup', default=16, type=int)
parser.add_argument('--T', default=5, type=int)
parser.add_argument('--tie_weight', default=False, type=str2bool)
parser.add_argument('-j', '--workers', default=16, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr_mode', default='step', type=str)
parser.add_argument('--warmup_mode', default='linear', type=str)
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--base_lr', '--learning-rate', default=0.1, type=float)
parser.add_argument('--step', default='30, 60, 90', type=str)
parser.add_argument('--decay-factor', default=0.1, type=float)
parser.add_argument('--decay-epoch', default=30, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
parser.add_argument('--print-freq', '-p', default=10, type=int)
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--resume-opt', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--distribute', action='store_true',
                    help='use slurm distributed training')
parser.add_argument('--port', default='23456', type=str)
parser.add_argument('--save-path', default='checkpoint', type=str)

best_prec1 = 0
best_prec5 = 0


def main():
    global args, best_prec1, best_prec5
    global rank, world_size
    args = parser.parse_args()
    if args.distribute:
        import multiprocessing as mp
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        rank, world_size = dist_init(args.port)
    else:
        rank, world_size = 0, 1
    assert(args.batch_size % world_size == 0)
    assert(args.workers % world_size == 0)
    args.batch_size = args.batch_size // world_size
    args.workers = args.workers // world_size

    if rank == 0:
        if not os.path.isdir(os.path.dirname(args.save_path)):
            os.makedirs(os.path.dirname(args.save_path))

    # sw config
    sw_cfg = dict(type='SW',
                  sw_type=args.sw_type,
                  num_pergroup=args.num_pergroup,
                  T=args.T,
                  tie_weight=args.tie_weight,
                  momentum=0.9,
                  affine=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('inception'):
        print('inception_v3 without aux_logits!')
        image_size = 341
        input_size = 299
        model = models.__dict__[args.arch](pretrained=args.pretrain)
    else:
        image_size = 256
        input_size = 224
        model = models.__dict__[args.arch](
                    pretrained=args.pretrain,
                    sw_cfg=sw_cfg if args.use_sw else None
                )

    if rank == 0:
        print(model)
        print('    Total params: %.2fM' %
              (sum(p.numel() for p in model.parameters())/1000000.0))
    model.cuda()
    if args.distribute:
        model = DistModule(model)
    else:
        model = torch.nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.load_path:
        if args.resume_opt:
            best_prec1, best_prec5, args.start_epoch = utils.load_state(
                args.load_path, model, optimizer=optimizer)
        else:
            utils.load_state(args.load_path, model)
        torch.cuda.empty_cache()

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = DistributedSampler(train_dataset) if args.distribute else None
    val_sampler = DistributedSampler(val_dataset) if args.distribute else None

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False if args.distribute else True,
                              num_workers=args.workers,
                              pin_memory=False,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=False,
                            sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    niters = len(train_loader)
    lr_scheduler = utils.LRScheduler(optimizer, niters, args)

    if rank == 0:
        tb_logger = SummaryWriter(args.save_path+'/events')
    else:
        tb_logger = None

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        prec1_train, loss_train = train(train_loader, model, criterion,
                                        optimizer, lr_scheduler, epoch)

        # evaluate on validation set
        prec1, prec5, loss_val = validate(val_loader, model, criterion)

        if rank == 0:
            # tb
            tb_logger.add_scalar('loss_train', loss_train, epoch)
            tb_logger.add_scalar('acc1_train', prec1_train, epoch)
            tb_logger.add_scalar('loss_test', loss_val, epoch)
            tb_logger.add_scalar('acc1_test', prec1, epoch)
            # remember best prec@1 and save checkpoint
            is_best1 = prec1 > best_prec1
            is_best5 = prec5 > best_prec5
            best_prec1 = max(prec1, best_prec1)
            best_prec5 = max(prec5, best_prec5)
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'prec1': prec1,
                'prec5': prec5,
                'optimizer': optimizer.state_dict(),
            }, is_best1, is_best5, args.save_path + '/model')


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr = lr_scheduler.update(i, epoch)

        target = target.cuda()
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        # measure accuracy and record loss
        loss = criterion(output, target_var) / world_size
        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

        reduced_loss = loss.data.clone()
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec5 = prec5.clone() / world_size

        if args.distribute:
            dist.all_reduce_multigpu([reduced_loss])
            dist.all_reduce_multigpu([reduced_prec1])
            dist.all_reduce_multigpu([reduced_prec5])

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(reduced_prec1.item(), input.size(0))
        top5.update(reduced_prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        if args.distribute:
            average_gradients(model)
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            print('Ep: [{0}][{1}/{2}]  '
                  'T {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'D {data_time.val:.2f} ({data_time.avg:.2f})  '
                  'LR {lr:.4f}  '
                  'L {loss.val:.3f} ({loss.avg:.4f})  '
                  'P1 {top1.val:.3f} ({top1.avg:.3f})  '
                  'P5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, lr=lr, loss=losses, top1=top1, top5=top5))
    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)

            # measure accuracy and record loss
            loss = criterion(output, target_var) / world_size
            prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

            reduced_loss = loss.data.clone()
            reduced_prec1 = prec1.clone() / world_size
            reduced_prec5 = prec5.clone() / world_size

            if args.distribute:
                dist.all_reduce_multigpu([reduced_loss])
                dist.all_reduce_multigpu([reduced_prec1])
                dist.all_reduce_multigpu([reduced_prec5])

            losses.update(reduced_loss.item(), input.size(0))
            top1.update(reduced_prec1.item(), input.size(0))
            top5.update(reduced_prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and rank == 0:
                print('Test: [{0}/{1}]  '
                      'T {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                      'L {loss.val:.3f} ({loss.avg:.4f})  '
                      'P1 {top1.val:.3f} ({top1.avg:.3f})  '
                      'P5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        if rank == 0:
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    main()
