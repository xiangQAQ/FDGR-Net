import torch.nn as nn
import math
import torch


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):
    def __init__(self, version=1.0):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self._conv1 = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3)
        self._relu1 = nn.ReLU(inplace=True)
        self._maxp1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self._fire1 = Fire(96, 16, 64, 64)
        self._fire2 = Fire(128, 16, 64, 64)
        self._fire3 = Fire(128, 32, 128, 128)
        self._maxp2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self._fire4 = Fire(256, 32, 128, 128)
        self._fire5 = Fire(256, 48, 192, 192)
        self._fire6 = Fire(384, 48, 192, 192)
        self._fire7 = Fire(384, 64, 256, 256)
        self._maxp3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self._fire8 = Fire(512, 64, 256, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        x = self._conv1(x)
        x1 = self._relu1(x)
        x = self._maxp1(x1)
        x = self._fire1(x)
        x = self._fire2(x)
        x2 = self._fire3(x)
        x = self._maxp2(x2)
        x = self._fire4(x)
        x = self._fire5(x)
        x = self._fire6(x)
        x3 = self._fire7(x)
        x = self._maxp3(x3)
        x4 = self._fire8(x)
        return [x4, x3, x2, x1]