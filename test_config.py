import os
import os.path
import sys
import numpy as np

        
class Config:
    backbone_name = 'DenseNet169'
    num_class = 19
    test_img_path = os.path.join('data', 'head', 'RawImage', 'Test2Data')
    test_gt_path = os.path.join('data', 'head', 'head_test.json')
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    data_shape = (896, 768)
    output_shape = (224, 192)

    gk7 = (9, 9)

    use_GT_bbox = True

cfg = Config()
