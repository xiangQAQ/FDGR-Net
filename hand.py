import os
import os.path
import sys
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
class Config:

    model = 'CPN50' # option 'CPN50', 'CPN101'

    num_class = 29
    test_img_path = os.path.join('data', 'hand', 'hand_test_img')
    test_gt_path = os.path.join('data', 'hand', 'hand_test.json')

    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
                (23, 24), (25, 26), (27, 28)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    data_shape = (512, 384)
    output_shape = (128, 96)

    gk7 = (7, 7)

    use_GT_bbox = True

cfg = Config()