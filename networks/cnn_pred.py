import torch.nn as nn
import torch
import math
from networks.new_conv.condconv import CondConv2D
from networks.new_conv.ASconv import ASConv
import torch.nn.functional as F


class Cnn_predNet(nn.Module):
    def __init__(self, channel_settings):
        super(Cnn_predNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples = [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))

        return nn.Sequential(*layers)


    def forward(self, x):
        global_fms = []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
        return global_fms


class fs_layer(nn.Module):
    def __init__(self, num_class):
        super(fs_layer, self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = ASConv(65, 64, 7, 1, padding=3, groups=1, K=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.fs = self._fs(num_class)


    def forward(self, image, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        image = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
        x = torch.cat([x, image], dim=1)
        x = self.relu2(self.bn2(self.conv2(x)))

        global_fms = []
        for n in range(self.num_class):
            global_fms.append(self.fs[n](x))
        return global_fms

    def _fs(self, num_class):
        layers = []
        for i in range(num_class):
            layers.append(nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)))
        return nn.ModuleList(layers)

class Cnn_featspr(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(Cnn_featspr, self).__init__()
        self.channel_settings = channel_settings
        self.num_class = num_class
        predict = []
        for i in range(len(channel_settings)-1):
            predict.append(self._predict(output_shape, num_class))
        self.predict = nn.ModuleList(predict)
        self.fs_layer = fs_layer(num_class)
        self.fs_predict = self._fs_predict(num_class)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def _fs_predict(self, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 64,
                                kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, num_class,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)


    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 64,
                                kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, num_class,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(num_class))
        return nn.Sequential(*layers)

    def forward(self, image, x):
        global_outs = []
        for i in range(len(self.channel_settings)):
            if i != len(self.channel_settings)-1:
                outs = self.predict[i](x[i])
                global_outs.append(outs)
            else:
                outs = self.fs_predict(x[i])
                global_outs.append(outs)
                global_fms = self.fs_layer(image, x[i])
        return global_fms, global_outs
