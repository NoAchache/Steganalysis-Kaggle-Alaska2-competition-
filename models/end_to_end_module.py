import torch.nn as nn
import torch
# https://github.com/lukemelas/EfficientNet-PyTorch
from efficientnet_pytorch import EfficientNet

from models.network import SrmNetwork


class EndToEndNet(nn.Module):
    def __init__(self):
        super(EndToEndNet, self).__init__()
        self.srm = SrmNetwork()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b2', in_channels=30)
        self.efficient_net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, img_input):
        srm_image = self.srm(img_input)
        eff_out = self.efficient_net(srm_image)
        return self.log_softmax(eff_out)


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.nll_loss = torch.nn.NLLLoss()

    def forward(self, classification_output, label):
        classif_loss = self.nll_loss(classification_output, label)

        return classif_loss