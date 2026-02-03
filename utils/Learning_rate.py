# -*- coding = utf-8 -*-
# @Time: 2025/10/20 15:11
# @Author: wisehone
# @File: Learning_rate.py
# @SoftWare: PyCharm
import math


def adjust_learning_rate(optimizer, epoch, args):
    lr_adjust = {}
    if args.Iradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.Iradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.Iradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.5 ** ((epoch - 3) // 1))}
    elif args.Iradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.Iradj in ['cosine', 'card']:
        # warm-up
        min_lr = 0
        warmup_epoches = 0
        lr = (min_lr + (args.learning_rate - min_lr)) * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epoches) / (args.train_epochs - warmup_epoches)))
        lr_adjust = {epoch: lr}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))