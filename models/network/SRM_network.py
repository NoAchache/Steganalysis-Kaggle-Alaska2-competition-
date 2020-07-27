import torch.nn as nn
import torch

from models.network.truncation_layers import TLU, STL


class SrmNetwork(nn.Module):
    def __init__(self):
        super(SrmNetwork, self).__init__()

        self.conv3x3 = nn.Conv2d(3, 17, 3, 1, padding=1)
        self.conv5x5 = nn.Conv2d(3, 13, 5, 1, padding=2)
        self.bn = torch.nn.BatchNorm2d(30)

        self.tlu = TLU(T=8 / 255)
        # self.stl = STL(T=8 / 255)


    def forward(self, x):
        c3x3 = self.conv3x3(x)
        c5x5 = self.conv5x5(x)
        c = torch.cat((c3x3, c5x5), 1)
        c = self.bn(self.tlu(c))
        return c


