"""This module provides the paths for all original and whitened
image data.
"""

import os
import sys

if __file__:
    file_path = os.path.realpath(__file__)
    ROOT_DIR = os.path.dirname(os.path.dirname(file_path))
    print("Accessing root modules...")
    sys.path.append(ROOT_DIR)
else:
    print(
        "File run in interactive environment.",
        "Cannot append ROOT_DIR to sys.path..."
    )

from utilities.root_dir import ROOT_DIR
from utilities.path_generator import get_data_path


# Paths for all original data
training_images_path = get_data_path(False, 'training', False)
training_labels_path = get_data_path(False, 'training', True)

validation_images_path = get_data_path(False, 'validation', False)
validation_labels_path = get_data_path(False, 'validation', True)

test_images = get_data_path(False, 'test')

# Paths for all whitened data
w_training_images_path = get_data_path(True, 'training', False)
w_training_labels_path = get_data_path(True, 'training', True)

w_validation_images_path = get_data_path(True, 'validation', False)
w_validation_labels_path = get_data_path(True, 'validation', True)

w_test_images = get_data_path(True, 'test')

# For debugging only:
if __name__ == '__main__':
    print(training_images_path)
    print(training_labels_path)
    print(validation_images_path)
    print(validation_labels_path)

    print(test_images)

    print(w_training_images_path)
    print(w_training_labels_path)
    print(w_validation_images_path)
    print(w_validation_labels_path)

    print(w_test_images)