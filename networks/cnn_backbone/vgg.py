import torch
import torch.nn as nn


def CBR(in_channels ,out_channels):

    cbr =nn.Sequential(

        nn.Conv2d(in_channels, out_channels, kernel_size=3 ,stride=1 ,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return cbr


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        block_nums = [2, 2, 3, 3, 3]
        self.block1 = self._make_layers(in_channels=1, out_channels=64, block_num=block_nums[0])
        self.block2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.block3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.block4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.block5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])

    def _make_layers(self, in_channels, out_channels, block_num):
        blocks = []
        blocks.append(CBR(in_channels, out_channels))

        for i in range(1, block_num):
            blocks.append(CBR(out_channels, out_channels))

        blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.block1(x)
        x1 = self.block2(x)
        x2 = self.block3(x1)
        x3 = self.block4(x2)
        x4 = self.block5(x3)
        return [x4, x3, x2, x1]

"""
input_img = torch.rand(1, 1, 224, 224)
block_nums = [2, 2, 3, 3, 3]
model = VGG()
PN = sum(p.numel() for p in model.parameters())
out = model(input_img)
xx=1
"""
