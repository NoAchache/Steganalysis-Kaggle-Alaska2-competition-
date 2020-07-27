import torch
from torch import nn as nn

from utils import cfg


class TLU(nn.Module):
    '''
    Truncated Linear Unit
    '''
    def __init__(self, T):
        super(TLU, self).__init__()
        self.T = T

    def forward(self, x):
        x = torch.clamp(x, -self.T, self.T)
        return x


class STL(nn.Module):
    '''
    Single-value Truncation Layer (STL) as proposed in CisNet: https://arxiv.org/pdf/1912.06540.pdf
    '''

    def __init__(self, T):
        super(STL, self).__init__()
        self.T = torch.tensor(T).to(cfg.device)

    def forward(self, x):
        x = torch.where(torch.abs(x) > self.T, self.T, x)
        return x