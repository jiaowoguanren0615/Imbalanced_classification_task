# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import torch
from timm.utils import accuracy
from .losses import FocalLoss
import util.utils as utils


def train_one_epoch(model: torch.nn.Module, criterion: FocalLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    set_training_mode=True, scalar=None):

    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if scalar is not None:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, scalar=None):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    for x_sample, y_sample in metric_logger.log_every(data_loader, 10, header):
        x_sample = x_sample.to(device, non_blocking=True)
        y_sample = y_sample.to(device, non_blocking=True)

        # compute output
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output = model(x_sample)
                loss = criterion(output, y_sample)
        else:
            output = model(x_sample)
            loss = criterion(output, y_sample)

        acc1, acc5 = accuracy(output, y_sample, topk=(1, 5))

        batch_size = x_sample.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    # print(output.mean().item(), output.std().item())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}