import torch.nn as nn
import torch
import math

def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out


class GFRMCell(nn.Module):
    def __init__(self):
        super(GFRMCell, self).__init__()
        self.gate1 = nn.Conv2d(64, 1,kernel_size=1, stride=1)
        self.gate2 = nn.Conv2d(64, 1,kernel_size=1, stride=1)

    def forward(self, x, x_, m):
        rt = self.gate1(x)
        mean = torch.mean(rt).detach()
        std = torch.std(rt).detach()
        rt = GaussProjection(rt, mean, std)
        rt = rt * x_

        at = self.gate2(x_)
        mean = torch.mean(at).detach()
        std = torch.std(at).detach()
        at = GaussProjection(at, mean, std)
        at = at * x
        return x+rt-at

class ConvGFRM(nn.Module):
    def __init__(self, input_dim, num):
        super(ConvGFRM, self).__init__()
        self.input_dim = input_dim
        self.num = num
        cell_list = [GFRMCell()] * self.num
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, all_input):
        y_list = []
        for m in range(self.num):
            y = self.cell_list[m](all_input[m], sum(all_input[0:m]+all_input[m + 1:]), m)
            y_list.append(y)
        return y_list

class Inference_gate(nn.Module):
    def __init__(self, num_class):
        super(Inference_gate, self).__init__()
        self.num_class = num_class
        self.gru1 = ConvGFRM(input_dim=64, num=19)
        self.predict = self._fs_predict(self.num_class)


    def _fs_predict(self, num_class):
        layers = []
        for i in range(num_class):
            layers.append(nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(1)))
        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.gru1(x)
        outs = self._pred_forward(x)
        return outs

    def _pred_forward(self, x):
        outs = []
        for n in range(self.num_class):
            out = self.predict[n](x[n])
            outs.append(out)
        return torch.cat(outs, dim=1)
