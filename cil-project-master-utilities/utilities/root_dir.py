"""Utility function to define the project root consistently
"""

import os
import sys

if __file__:
    file_path = os.path.realpath(__file__)
    ROOT_DIR = os.path.dirname(os.path.dirname(file_path))
    print('ROOT_DIR is:', ROOT_DIR)
else:
    print('File run interactively. ROOT_DIR will be empty')
    ROOT_DIR = ''
