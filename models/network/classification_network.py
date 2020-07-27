import torch.nn as nn

# https://github.com/lukemelas/EfficientNet-PyTorch
from efficientnet_pytorch import EfficientNet


class ClassificationNetwork(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassificationNetwork, self).__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b2', in_channels=in_channels)
        self.efficient_net._fc = nn.Linear(in_features=1408, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.efficient_net(x)

