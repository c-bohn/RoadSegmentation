"""Simple utility function for generating system independent paths.
"""

import os
from utilities.root_dir import ROOT_DIR


def get_data_path(whitening, category, gt=False):
    p1 = os.path.join(ROOT_DIR, 'data', 'whitened' if whitening else 'original')
    p2 = os.path.join(p1, category)
    if category == 'test':
        p3 = p2
    else:
        p3 = os.path.join(p2, 'groundtruth' if gt else 'images')
    return os.path.join(p3, '')