import sys
import os

import warnings

from model import FCNet

from utils import save_checkpoint, gt_orth_loss, gt_sim_loss, get_sep_loss,get_sim_loss

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch FCNet')

parser.add_argument('--train_json', type=str, default='./datasets/density_train_list.json', metavar='TRAIN',
                    help='path to train json')

parser.add_argument('--val_json', type=str, default='./datasets/density_val_list.json', metavar='VAL',
                    help='path to val json')

parser.add_argument('--test_json', type=str, default='./datasets/density_test_list.json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--output', type=str, default="./saved_modelss/", metavar='VAL',
                    help='path output')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('--best', '-b', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('--initial', '-i', metavar='PRETRAINED', default=None, type=int,
                    help='path to the pretrained model')
tf = open("image_class.json", "r")
class_dict = json.load(tf)

def main():
    global args, best_prec1

    best_prec1 = 1e6
    args = parser.parse_args()
    args.lr = 1e-4
    args.batch_size = 8  # 26
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 200
    args.workers = 4
    args.seed = int(time.time())
    args.print_freq = 50

    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)

    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)

    torch.cuda.manual_seed(args.seed)

    model = FCNet()

    model = model.cuda()

    criterion = nn.MSELoss(reduction='sum').cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.decay)


    ###########
    if args.best:
        print("=> loading best checkpoint '{}'".format(args.best))

        checkpoint = torch.load(os.path.join(args.output, 'model_best.pth.tar'))

        model.load_state_dict(checkpoint['state_dict'])

        best_prec1 = validate(val_list, model, criterion)

        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    if args.initial:
        args.start_epoch = args.initial
        print(args.initial)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_list, model, criterion, optimizer, epoch)

        prec1 = validate(val_list, model, criterion)

        is_best = prec1 < best_prec1

        best_prec1 = min(prec1, best_prec1)

        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.output, filename='checkpoint.pth.tar')

    print('Train process finished')


def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            seen=model.seen,
                            batch_size=args.batch_size,
                            num_workers=args.workers),
        batch_size=args.batch_size)

    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (img, target, img_path) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        img = Variable(img)
        uu, zz, output = model(img, "train")
        output = output[:,0,:,:]

        target = target.type(torch.FloatTensor).cuda()
        target = Variable(target)

        count_loss = criterion(output, target)
#         orth_loss = gt_orth_loss(uu[0,:,:,:].unsqueeze(0), zz)
        sep_loss = get_sep_loss(uu[0,:,:,:].unsqueeze(0), zz)
    
        loss = count_loss + sep_loss

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            print(count_loss)
            print(sep_loss)


def validate(val_list, model, criterion):
    print('begin val')
    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        batch_size=1)

    model.eval()

    mae = 0

    for i, (img, target, img_path) in enumerate(val_loader):
        img = Variable(img.cuda())
        density = model(img, "").data.cpu().numpy()
        pred_sum = density.sum()
        print("pt:",pred_sum,"gt:",target.sum())
        mae += abs(pred_sum - target.sum())

    mae = mae / len(val_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))

    return mae


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FCNet')

    parser.add_argument('--train_json', type=str, default='./datasets/density_train_list.json', metavar='TRAIN',
                        help='path to train json')

    parser.add_argument('--val_json', type=str, default='./datasets/density_val_list.json', metavar='VAL',
                        help='path to val json')

    parser.add_argument('--test_json', type=str, default='./datasets/density_test_list.json', metavar='TEST',
                        help='path to test json')

    parser.add_argument('--output', type=str, default="./saved_modelss/", metavar='VAL',
                        help='path output')

    parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                        help='path to the pretrained model')

    parser.add_argument('--best', '-b', metavar='PRETRAINED', default=None, type=str,
                        help='path to the pretrained model')

    parser.add_argument('--initial', '-i', metavar='PRETRAINED', default=None, type=int,
                        help='path to the pretrained model')

    main()
