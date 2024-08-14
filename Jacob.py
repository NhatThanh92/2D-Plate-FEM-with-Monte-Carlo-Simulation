# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:54:25 2024

@author: thanh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Jacobian(nodeCoord,nderiv):
    J = np.matmul(np.transpose(nodeCoord),nderiv) # matrix multiplication
    
    bT = np.transpose(nderiv) #(2x4)
    aT = J.T
    xT = np.linalg.solve(aT,bT)
    xyDeriv = np.transpose(xT)
    return J, xyDeriv # xyDeriv mean derivative x,y in strain-displacement matrix B