
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

def prob2Ent(prob):
    n, c = prob.size()
    ent = (-torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c))

    return ent

def get_current_consistency_weight(weight, epoch, rampup):
    """Consistency ramp-up from https://arxiv.org/abs/1610.02242"""
    return weight * sigmoid_rampup(epoch, rampup)


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

class OldWeightEMA(object):
    """Exponential moving average weight optimizer for mean teacher model"""

    def __init__(self, ema_net, stu_net, ema_alpha):
        self.BaseNet = stu_net
        self.BaseNet_ema = ema_net
        self.alpha = ema_alpha

        for param_q, param_k in zip(self.BaseNet.parameters(), self.BaseNet_ema.parameters()):
            param_k.data = param_q.data.clone()
        for buffer_q, buffer_k in zip(self.BaseNet.buffers(), self.BaseNet_ema.buffers()):
            buffer_k.data = buffer_q.data.clone()

        ema_net.eval()

    def step(self):
        one_minus_alpha = 1.0 - self.alpha

        for param_q, param_k in zip(self.BaseNet.parameters(), self.BaseNet_ema.parameters()):
            param_k.data = param_k.data.clone() * self.alpha + param_q.data.clone() * one_minus_alpha

        for buffer_q, buffer_k in zip(self.BaseNet.buffers(), self.BaseNet_ema.buffers()):
            buffer_k.data = buffer_q.data.clone()


def weightmap(pred1, pred2):
    output = 1.0 - torch.sum((pred1 * pred2), 1).view(pred1.size(0), 1) / \
             (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(pred1.size(0), 1)
    return output

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [N, C]
        dst: target points, [M, C]
    Output:
        dist: per-point square distance, [N, M]
    """
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1, 0))
    dist += torch.sum(src ** 2, -1).view(N, 1)
    dist += torch.sum(dst ** 2, -1).view(1, M)

    return dist