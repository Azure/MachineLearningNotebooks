# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import numpy as np


def get_alphas():
    # list of numbers from 0.0 to 1.0 with a 0.05 interval
    return np.arange(0.0, 1.0, 0.05)
