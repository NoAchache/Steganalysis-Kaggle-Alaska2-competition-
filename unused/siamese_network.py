# Adaptation of the implementation of the orginal paper taken from : https://github.com/fangpin/siamese-pytorch
# Changes of the sizes of the kernels (use of smaller filters since stego noise is fastly varying) + use of average
# pooling instead max pooling as suggested in the literature for steganalysis
import torch.nn as nn
import torch

class SiameseNetwork(nn.Module):
    def __init__(self, num_channels):
        super(SiameseNetwork, self).__init__()

        self.conv = nn.Sequential(
            # 16*16
            nn.Conv2d(num_channels, 160, 3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(160, 320, 3, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 8*8
            nn.Conv2d(320, 640, 3, padding=1),
            nn.BatchNorm2d(640),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 4*4
            nn.Conv2d(640, 640, 3, padding=1),
            nn.BatchNorm2d(640),
            nn.ReLU()
        )
        self.linear = nn.Sequential(nn.Linear(10240, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out, self.sigmoid(out)

