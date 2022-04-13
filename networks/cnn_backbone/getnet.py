from .densenet import densenet121, densenet169
from .squeezenet import SqueezeNet
from .vgg import VGG
from .efficientnet import EfficientNet
from .resnet import *

def get_backnet(backbone_name, pretrained):
    if backbone_name == 'ResNet18':
        channel_settings = [512, 256, 128, 64]
        backbone = resnet18(pretrained=pretrained)
    elif backbone_name == 'ResNet34':
        channel_settings = [512, 256, 128, 64]
        backbone = resnet34(pretrained=pretrained)
    elif backbone_name == 'ResNet50':
        channel_settings = [2048, 1024, 512, 256]
        backbone = resnet50(pretrained=pretrained)
    elif backbone_name == 'DenseNet121':
        channel_settings = [1024, 1024, 512, 256] # 64 32
        backbone = densenet121()
    elif backbone_name == 'DenseNet169':
        channel_settings = [1664, 1280, 512, 256] # 64 32
        backbone = densenet169()
    elif backbone_name == 'SqueezeNet':
        channel_settings = [512, 512, 256, 96]
        backbone = SqueezeNet()
    elif backbone_name == 'EfficientNet-b0':
        channel_settings = [1280, 112, 40, 24]
        backbone = EfficientNet.from_name('efficientnet-b0')
    elif backbone_name == 'EfficientNet-b1':
        channel_settings = [1280, 112, 40, 24]
        backbone = EfficientNet.from_name('efficientnet-b1')
    elif backbone_name == 'VGG':
        channel_settings = [512, 512, 256, 128]
        backbone = VGG()

    return channel_settings, backbone