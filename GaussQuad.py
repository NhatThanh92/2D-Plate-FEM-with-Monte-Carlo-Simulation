# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:53:30 2024

@author: thanh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gaussQuadrature(option):
    if option == 1: # full integration ( 4 Gauss points)
        o = 1/np.sqrt(3)
        locations = np.array([[-o,-o],
                              [o,-o],
                              [o,o],
                              [-o,o]])
        weights = np.ones((4,1))
    else: # (1 Gauss point)
        locations = np.zeros((1,2))
        weights = 4
    return weights, locations