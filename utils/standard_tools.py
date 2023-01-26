import numpy as np
import torch

"""
def recursiveToTensor(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = recursiveToTensor(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [recursiveToTensor(x) for x in data]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(data, bool):
        data = torch.tensor(data)
    return data
"""

def recursiveToTensor(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = recursiveToTensor(data[key])
    elif isinstance(data, list) or isinstance(data, tuple):
        data = [recursiveToTensor(x) for x in data]
        # data = torch.tensor(data)
    elif isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data).float()
    # if isinstance(data, bool):
    #     data = torch.tensor(data)
    elif  torch.is_tensor(data):
        return data
    # else:
    #     data = torch.tensor(data)
    return data

def togpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [togpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:togpu(_data) for key,_data in data.items()}
    # else:
    #     data = torch.tensor(data)
    else:
        if not torch.is_tensor(data):
            data = torch.tensor(data)
        data = data.contiguous().cuda(non_blocking=True)
    return data

def tolong(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = tolong(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [tolong(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data

def recursiveToNumpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = recursiveToTensor(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = np.array(data)
    return data