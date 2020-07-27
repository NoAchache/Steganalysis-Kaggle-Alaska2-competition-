# 'Translated' from Keras: https://github.com/nsarang/ImageDenoisingAutoencdoer/blob/master/DenoisingAutoencoder.ipynb

import torch.nn as nn
import torch


class ConvBnPRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(ConvBnPRelu, self).__init__()

        padding = kernel // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class TransposeConvBnPRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(TransposeConvBnPRelu, self).__init__()

        padding = kernel // 2
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding=padding,
                                       output_padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, transp_in_channels, transp_out_channels, transp_kernel, conv_in_channels, conv_out_channels,
                 conv_kernel):
        super(DeconvBlock, self).__init__()
        self.transp_layer = TransposeConvBnPRelu(transp_in_channels, transp_out_channels, transp_kernel, 2)
        if conv_in_channels is not None:
            self.conv_layer = ConvBnPRelu(conv_in_channels, conv_out_channels, conv_kernel, 1)
        else:
            self.conv_layer = None

    def forward(self, encoding_layer, decoding_layer):
        x = self.transp_layer(decoding_layer)
        x = torch.cat((x, encoding_layer), 1)
        if self.conv_layer is not None:
            x = self.conv_layer(x)
        return x


class DenoisingAutoencoder(nn.Module):
    def __init__(self, num_channels):
        super(DenoisingAutoencoder, self).__init__()

        self.num_channels = num_channels
        self.encoding1 = ConvBnPRelu(in_channels=self.num_channels, out_channels=160, kernel=3, stride=1)
        self.encoding2 = ConvBnPRelu(in_channels=160, out_channels=160, kernel=3, stride=2)
        self.encoding3 = ConvBnPRelu(in_channels=160, out_channels=320, kernel=3, stride=1)
        self.encoding4 = ConvBnPRelu(in_channels=320, out_channels=640, kernel=5, stride=2)
        self.encoding5 = ConvBnPRelu(in_channels=640, out_channels=640, kernel=3, stride=1)

        self.decoding1 = DeconvBlock(transp_in_channels=640, transp_out_channels=640, transp_kernel=3,
                                     conv_in_channels=960, conv_out_channels=320, conv_kernel=3)
        self.decoding2 = DeconvBlock(transp_in_channels=320, transp_out_channels=160, transp_kernel=3,
                                     conv_in_channels=None, conv_out_channels=None, conv_kernel=None)

        self.predict = nn.Conv2d(320, self.num_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.encoding1(x)
        e2 = self.encoding2(e1)
        e3 = self.encoding3(e2)
        e4 = self.encoding4(e3)
        e5 = self.encoding5(e4)

        d1 = self.decoding1(e3, e5)
        d2 = self.decoding2(e1, d1)


        pred = self.predict(d2)
        restored_image = self.sigmoid(pred)

        return restored_image
