'''Misc module:
    Miscellaneous helper functions and utilities.
'''

import os
import glob


# Helper function to find a file or folder path
def find_path(name, path_prefix):
    for root, _, _ in os.walk(path_prefix):
        if glob.glob(os.path.join(root, name)):
            return root
