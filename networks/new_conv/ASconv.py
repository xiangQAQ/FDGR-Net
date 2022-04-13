import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_weight(out_planes,in_planes,kernel_size):
    weight1 = torch.randn(1, out_planes, in_planes, kernel_size, kernel_size)
    weight1[:, :, :, 0:2, :] = 0
    weight1[:, :, :, :, 0:2] = 0
    weight1[:, :, :, 5:7, :] = 0
    weight1[:, :, :, :, 5:7] = 0

    weight2 = torch.randn(1, out_planes, in_planes, kernel_size, kernel_size)
    weight2[:, :, :, 0, :] = 0
    weight2[:, :, :, 2, :] = 0
    weight2[:, :, :, 4, :] = 0
    weight2[:, :, :, 6, :] = 0
    weight2[:, :, :, :, 0] = 0
    weight2[:, :, :, :, 2] = 0
    weight2[:, :, :, :, 4] = 0
    weight2[:, :, :, :, 6] = 0

    weight3 = torch.randn(1, out_planes, in_planes, kernel_size, kernel_size)
    weight3[:, :, :, 1, :] = 0
    weight3[:, :, :, 2, :] = 0
    weight3[:, :, :, 4, :] = 0
    weight3[:, :, :, 5, :] = 0
    weight3[:, :, :, :, 1] = 0
    weight3[:, :, :, :, 2] = 0
    weight3[:, :, :, :, 4] = 0
    weight3[:, :, :, :, 5] = 0

    return torch.cat((weight1,weight2,weight3), dim=0)



class Attention(nn.Module):
    def __init__(self,in_planes,K):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.net=nn.Conv2d(in_planes, K, kernel_size=1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        att=self.avgpool(x)
        att=self.net(att)
        att=att.view(x.shape[0],-1)
        return self.sigmoid(att)


class ASConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0,
                 groups=1,K=4):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.K = K
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = Attention(in_planes=in_planes,K=K)
        self.weight =  nn.Parameter(get_weight(out_planes, in_planes, kernel_size),requires_grad=True)

    def forward(self,x):
        N,in_planels, H, W = x.shape
        softmax_att=self.attention(x)
        x=x.view(1, -1, H, W)

        weight = self.weight
        weight = weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_att,weight)
        aggregate_weight = aggregate_weight.view(
            N*self.out_planes, self.in_planes//self.groups,
            self.kernel_size, self.kernel_size)
        output=F.conv2d(x,weight=aggregate_weight,
                        stride=self.stride, padding=self.padding,
                        groups=self.groups*N)
        output=output.view(N, self.out_planes, H, W)
        return output



"""
indata = torch.rand(4, 1, 64, 64)
net = ASConv(1, 32, 7, 1, padding=3, groups=1, K=3)
outdata = net(indata)
xx=1
"""