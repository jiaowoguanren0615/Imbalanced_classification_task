import argparse
import datetime
import pandas as pd
import numpy as np
import time, os, re
from sklearn.model_selection import train_test_split
import torch
import torch.backends.cudnn as cudnn
import json
import torch.nn as nn
from pathlib import Path
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from util import *
from retnet import RetNet_tiny, RetNet_small_V1, RetNet_small_V2, RetNet_large, RetNet_huge
from estimate_model import Predictor, Plot_ROC


def create_model(input_features, num_classes, **kwargs):
    model = RetNet_tiny()
    hidden_dim = model.hidden_dim

    class RetNetModel(nn.Module):
        def __init__(self, input_features, num_classes):
            super(RetNetModel, self).__init__()
            self.first_linear = nn.Linear(input_features, hidden_dim)
            self.feature_extra = model
            self.pre_logits = nn.Identity()
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward_features(self, x):
            x = self.first_linear(x).unsqueeze(1)
            x = self.feature_extra(x)
            return self.pre_logits(x[:, 0])

        def forward(self, x):
            x = self.forward_features(x)
            x = self.fc(x)
            return x

    net = RetNetModel(input_features, num_classes)
    return net


def build_dataset(args):
    if re.findall('.csv', args.data_root):
        dfMerge = pd.read_csv(args.data_root)
    elif re.findall('.txt', args.data_root):
        dfMerge = pd.read_table(args.data_root)
    elif re.findall('.xlsx|.xls', args.data_root):
        dfMerge = pd.read_excel(args.data_root)
    else:
        raise ValueError(
            'Your file does not belongs to one of csv txt xlsx, please modify the process reading data by yourself')

    dfMerge = dfMerge.drop(['#user_id'], axis=1)

    dfMerge['tag_value_bool'] = dfMerge['tag_value_bool'].map({True: 'T', False: 'F'})
    classes = dfMerge['tag_value_bool'].unique()

    class_indices = {k: v for v, k in enumerate(classes)}
    dfMerge['tag_value_bool'] = dfMerge['tag_value_bool'].map(class_indices)
    json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)
    with open('./classes_indices.json', 'w') as json_file:
        json_file.write(json_str)
        
    X = np.array(dfMerge.drop(['tag_value_bool'], axis=1))
    y = np.array(dfMerge['tag_value_bool'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2023)
    
    train_set = MyDataset(X_train, y_train)
    valid_set = MyDataset(X_test, y_test)
    return train_set, valid_set


def get_args_parser():
    parser = argparse.ArgumentParser(
        'RetNet training and evaluation script', add_help=False)
    parser.add_argument('--epochs', default=5, type=int)

    # Dataset parameters
    parser.add_argument('--data_root', default='./user_features_4w.csv', type=str,
                        help='dataset path')
    parser.add_argument('--batch-size', default=256, type=int)  # consume 3G GPU-Memory
    parser.add_argument('--input_features', default=33, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--output_dir', default='./save', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='./save/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_memory', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    parser.add_argument('--sync-bn', default=False, help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=0.01, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.025,
                        help='weight decay (default: 0.025)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 2e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_val = build_dataset(args=args)

    num_tasks = utils.get_world_size()

    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False
    )


    model = create_model(input_features=args.input_features, num_classes=args.num_classes)

    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False)
        model_without_ddp = model.module

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 1024.0
    args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = FocalLoss(alpha=0.4)

    output_dir = Path(args.output_dir)

    max_accuracy = 0.0
    if args.resume and os.path.exists('./save/checkpoint.pth'):
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        max_accuracy = checkpoint['max_accuracy']
        print(f'Max Accuracy is {max_accuracy:.2f} now!!')
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            args.start_epoch = checkpoint['epoch'] + 1

            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, criterion)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # scalar = torch.cuda.amp.GradScaler() if torch.cuda.is_bf16_supported() else None
    scalar = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode,
            scalar=scalar
        )

        lr_scheduler.step(epoch)

        test_stats = evaluate(data_loader_val, model, device, criterion, scalar=scalar)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if args.output_dir:
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')
            utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'max_accuracy': max_accuracy,
                    'args': args,
            },  f'./save/checkpoint_{epoch+1}.pth')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # predict and plot ROC
    print('*******************STARTING PREDICT*******************')
    Predictor(model_without_ddp, data_loader_val, './save/checkpoint_5.pth', device)
    Plot_ROC(model_without_ddp, data_loader_val, './save/checkpoint_5.pth', device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'RetNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
