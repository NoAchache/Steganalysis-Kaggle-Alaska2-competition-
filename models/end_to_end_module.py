import torch.nn as nn
import torch

from models.network import SrmNetwork, ClassificationNetwork


class EndToEndNet(nn.Module):
    def __init__(self):
        super(EndToEndNet, self).__init__()
        self.srm = SrmNetwork()
        self.efficient_net = ClassificationNetwork(in_channels=30, num_classes=4)
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