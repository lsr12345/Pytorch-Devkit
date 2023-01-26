'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: PPT训练框架入口

example:

'''


import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


import argparse
import numpy as np
import random

import warnings
import yaml

from data.dataloader import Data_loader
from trainer_ddp import Trainer
from model.model_factory import Classify_Model, DB_Model, Segmentation_Model, Yolox_Model, \
    Crnn_Model, Solo_Model, ICTransformer, ReBiSeNet_Model, MaskFormer_Model
from utils.common import find_free_port

MODEL_SELECT = {'Classify': Classify_Model, 'DB': DB_Model, 'Seg': Segmentation_Model, 'YOLOX':Yolox_Model,
                'CRNN':Crnn_Model, 'SOLO':Solo_Model, 'ICT': ICTransformer, 'ReBiSe': ReBiSeNet_Model, 'MaskFormer': MaskFormer_Model}

def arg_parser():
    parser = argparse.ArgumentParser("train parser")
    parser.add_argument(
        "-e", "--eval_interval", type=int, default=1, help="eval interval"
    )
    parser.add_argument(
        "-s", "--save_interval", type=int, default=1, help="save interval"
    )
    parser.add_argument(
        "-v", "--visual_batch_interval", type=int, default=10, help="save interval"
    )
    parser.add_argument(
        "-ste", "--start_eval", type=int, default=0, help="save interval"
    )
    parser.add_argument(
        "-se", "--seed", type=int, default=None, help="random seed"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="GPU device for training"
    )
    parser.add_argument(
        "--nprocs", default=1, type=int, help="GPU device for training"
    )
    parser.add_argument(
        "--syncBN", default=False, action="store_true", help="syncBN"
    )
    parser.add_argument(
        "-c", "--ckpt", default=None, type=str, help="checkpoint file"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument(
        "-pre", "--pretrained", default=None, type=str, help="pretrained file"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default='./config/Config.yaml',
        type=str,
        help="training description file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default='./checkpoints',
        type=str,
        help="save dir",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )

    return parser

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  
        cudnn.deterministic = False
        cudnn.benchmark = True

def main():
    args = arg_parser().parse_args()
    args.nprocs = torch.cuda.device_count()

    args.distributed = True if args.nprocs > 1 else False
    args.dis_backend = 'nccl'

    dist_url = "tcp://127.0.0.1"
    port = find_free_port()
    args.dist_url = "{}:{}".format(dist_url, str(port) )

    with open(args.exp_file, mode='r') as fr:
        cfg = yaml.load(fr, Loader=yaml.FullLoader)

    if args.distributed:
        mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args, cfg))
    else:
        main_worker(args.local_rank, args.nprocs, args, cfg)

def main_worker(local_rank,nprocs, args, cfg):
    assert ( torch.cuda.is_available()), "cuda is not available. Please check your installation."
    args.rank = local_rank
    cfg['distributed'] = args.distributed
    init_seeds(local_rank+1)

    cudnn.benchmark = True
    if args.distributed:
        dist.init_process_group(backend=args.dis_backend,
                                                        init_method=args.dist_url,
                                                        world_size=nprocs,
                                                        rank=local_rank)

    Model = MODEL_SELECT[cfg['experiment_name']](config=cfg,  amp_training=args.fp16)
    DATA_Loader = Data_loader(config=cfg, args=args)

    trainer = Trainer(cfg, args, Model, DATA_Loader, step_update=True)
    trainer.train()

if __name__ == '__main__':
    main()