import os
import os.path
import sys
import numpy as np


        
class Config:
    # 'ResNet18','ResNet34', 'ResNet50', 'DenseNet121', 'DenseNet169', 'SqueezeNet', 'EfficientNet-b0', 'EfficientNet-b1', 'VGG'
    backbone_name = 'VGG'
    pretranined = True

    lr = 5e-4
    lr_gamma = 0.5
    lr_dec_epoch = list(range(20, 50, 10))

    batch_size = 4
    weight_decay = 1e-5


    num_class = 19
    train_img_path = os.path.join('data', 'head', 'RawImage', 'TrainingData')
    train_gt_path = os.path.join('data', 'head', 'head_train.json')
    test_img_path = os.path.join('data', 'head', 'RawImage', 'Test1Data')
    test_gt_path = os.path.join('data', 'head', 'head_val.json')
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    # data augmentation setting
    scale_factor=(0.85, 1.15)
    rot_factor=15

    data_shape = (896, 768)
    output_shape = (224, 192)
    gaussain_kernel = (9, 9)

    gk15 = (17, 17)
    gk11 = (15, 15)
    gk9 = (11, 11)
    gk7 = (9, 9)

cfg = Config()

