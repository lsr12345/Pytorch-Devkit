'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: PPT common functions

example:

'''
# coding: utf-8

import os
from functools import partial

import torch
import torch.distributed as dist

from loguru import logger

# # distribute function and config ##
# _LOCAL_PROCESS_GROUP = None

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def remove_file(file_dir, key_words=''):
    assert key_words != ''
    for fn in os.listdir(file_dir):
        if key_words in fn:
            os.remove(os.path.join(file_dir, fn))
            return True
    else:
        return False

def prepare_device(local_rank, local_world_size, distributed=False):
    '''
        setup GPU device if available, move model into configured device
    :param local_rank:
    :param local_world_size:
    :return:
    '''
    if distributed:
        ngpu_per_process = torch.cuda.device_count() // local_world_size
        device_ids = list(range(local_rank * ngpu_per_process, (local_rank + 1) * ngpu_per_process))

        if torch.cuda.is_available() and local_rank != -1:
            torch.cuda.set_device(device_ids[0])  # device_ids[0] =local_rank if local_world_size = n_gpu per node
            device = 'cuda'
        else:
            device = 'cpu'
        device = torch.device(device)
        return device, device_ids
    else:
        n_gpu = torch.cuda.device_count()
        n_gpu_use = local_world_size
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu

        list_ids = list(range(n_gpu_use))
        if n_gpu_use > 0:
            torch.cuda.set_device(list_ids[0])  # only use first available gpu as devices
            device = 'cuda'
        else:
            device = 'cpu'
        device = torch.device(device)
        return device, list_ids

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
            map the multiple outputs of the ``func`` into different
            list. Each list contains the same type of outputs corresponding
            to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

# def togpu(data, requires_grad=True):
#     """
#     Transfer tensor in `data` to gpu recursively
#     `data` can be dict, list or tuple
#     """
#     if isinstance(data, list) or isinstance(data, tuple):
#         data = [togpu(x) for x in data]
#     elif isinstance(data, dict):
#         data = {key:togpu(_data) for key,_data in data.items()}
#     elif isinstance(data, torch.Tensor):
#         data = data.contiguous().cuda(non_blocking=True).requires_grad = requires_grad
#     return data

# def togpu(data):
#     """
#     Transfer tensor in `data` to gpu recursively
#     `data` can be dict, list or tuple
#     """
#     if isinstance(data, list) or isinstance(data, tuple):
#         data = [togpu(x) for x in data]
#     elif isinstance(data, dict):
#         data = {key:togpu(_data) for key,_data in data.items()}
#     # else:
#     #     data = torch.tensor(data)
#     else:
#         if not torch.is_tensor(data):
#             data = torch.tensor(data)
#         data = data.contiguous().cuda(non_blocking=True)
#     return data

# def tolong(data):
#     if isinstance(data, dict):
#         for key in data.keys():
#             data[key] = tolong(data[key])
#     if isinstance(data, list) or isinstance(data, tuple):
#         data = [tolong(x) for x in data]
#     if torch.is_tensor(data) and data.dtype == torch.int16:
#         data = data.long()
#     return data
###############################