import os
import sys

""" 
Helper script for accessing modules at the project root level
from whithin child directories. Python modules in child directories
that are called directly via 'python my_module.py' have no access
to to their parents by default, so we append the parent directory
manually to sys path.

USAGE: Simply import this module as the first import of a module in
some immediate child directory that you might run as the main module.

You only need to do this if said module has dependencies in the parent
directory.
"""


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