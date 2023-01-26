'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

def norm_add_multi_loss(loss_dict):
    for i, key in enumerate(loss_dict.keys()):
        if i == 0:
            loss = loss_dict[key]
            n = loss_dict[key].detach()
        else:
            loss = loss +  loss_dict[key]/n

    return loss

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

from . import  sigmoid_focal_loss_cuda, SigmoidFocalLoss_cuda

class SigmoidFocalLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target, gamma=2.0, alpha=0.25):
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        loss = sigmoid_focal_loss_cuda.forward(input, target, num_classes,  gamma, alpha)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_input = sigmoid_focal_loss_cuda.backward(input, target, d_loss, num_classes, gamma, alpha)
        return d_input, None, None, None, None

sigmoid_focal_loss_ = SigmoidFocalLossFunction.apply

def sigmoid_focal_loss(pred,
                    target,
                    weight=None,
                    gamma=2.0,
                    alpha=0.25,
                    reduction='mean',
                    avg_factor=None):
    loss = sigmoid_focal_loss_(pred, target, gamma, alpha)

    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss




