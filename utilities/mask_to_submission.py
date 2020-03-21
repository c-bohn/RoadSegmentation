#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re

foreground_threshold = 0.7 # percentage of pixels = 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""

    #The labeling of the 16x16 image patches happens here by averaging over the outputs of our CNN,
    #i.e., our CNN needs to produce outputs of the same size as the ground truth data and not just output
    #labels for the 16x16-patches.

    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


if __name__ == '__main__':
    # before running adapt the following paths
    submission_filename = 'results/submission.csv'
    image_filenames = []
    images = os.listdir('results/bagged_predictions/')
    #for i in range(1, 51):
    for i in range(94):
        #image_filename = 'training/groundtruth/satImage_' + '%.3d' % i + '.png'
        image_filename = images[i]
        print(str(i+1) + ' :' + image_filename)
        image_filenames.append('results/bagged_predictions/' + image_filename)
    masks_to_submission(submission_filename, *image_filenames)
