import argparse
import datetime
import json
import numpy as np
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import os
from torchvision import datasets, transforms
import util.misc as misc
import models_net_mamba
from engine import evaluate, evaluate_speed_test
import torchvision  # 需要导入torchvision
from torchvision.transforms import RandomErasing

# 从 fine-tune.py 复制 DualViewDataset
class DualViewDataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img2, _   = super().__getitem__(index)   # 同图再采一次
        return (img, img2), label                # ((x1,x2), y)


def get_args_parser():
    parser = argparse.ArgumentParser('NetMamba fine-tuning for traffic classification', add_help=False)
    # 64
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    # Model parameters
    parser.add_argument('--model', default='flow_mamba_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--data_path', default='./ddos_datasets/flow_image', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output/finetune',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/finetune',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # spatio temporal features
    parser.add_argument('--if_stat', action='store_true',
                        help='whether to use statistical features')
    # evaluation settings
    parser.add_argument('--speed_test', action='store_true')
    parser.add_argument('--byte_length', default=1600, type=int)
    # 关键：添加 use_cross_mamba 参数
    parser.add_argument('--use_cross_mamba', action='store_true',
                        help='Use CrossMamba1D instead of standard Mamba blocks')

    return parser


def build_dataset(data_split, args):
    """从 fine-tune.py 复制的 build_dataset 函数，使用 DualViewDataset"""
    mean = [0.5]
    std = [0.5]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        RandomErasing(p=0.3, scale=(0.02, 0.05)),
        transforms.Normalize(mean, std),
        transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
    ])
    '''
    #原始transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    '''
    assert data_split in ["train", "test", "valid"]
    root = os.path.join(args.data_path, data_split)
    dataset = DualViewDataset(root, transform=transform)
    print(dataset)
    return dataset


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 使用 build_dataset 替代原来的 datasets.ImageFolder
    dataset_val = build_dataset(data_split="test", args=args)
    labels = dataset_val.classes
    print(dataset_val.class_to_idx)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # 关键修改：传递 use_cross_mamba 参数
    model = models_net_mamba.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        byte_length=args.byte_length,
        use_cross_mamba=args.use_cross_mamba,  # 添加这行
    )
    try:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        print("Load model checkpoint from: %s" % args.resume)
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")
        print("Evaluate the model from scratch.")
    model.to(device)

    if args.speed_test:
        # speed_test 也需要用 DualViewDataset
        dataset_train = build_dataset(data_split="train", args=args)
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        evaluate_speed_test(data_loader_train, model, device, args)
        return

    # test_stats = evaluate(data_loader_val, model, device)
    print(f"Batch size: {args.batch_size}")
    test_stats = evaluate(data_loader_val, model, device, if_stat=args.if_stat)
    for k, v in test_stats.items():
        print(f"Test {k}: {v}")
    with open(os.path.join(args.output_dir, 'test_stats.json'), 'w') as f:
        new_stats = {k: v.tolist() if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v for k, v in test_stats.items()}
        json.dump(new_stats, f, indent=2)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)