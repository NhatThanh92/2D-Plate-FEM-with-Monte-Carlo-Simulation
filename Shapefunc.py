# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:55:37 2024

@author: thanh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def shapeFuncQ4(xi,eta):
    
    # create 4 row, 1 column
    shape = np.zeros((4,1)) 
    
    # : take all elements along this axis (all rows); 0: take element at index 0 (first column)
    shape[:,0] = 1/4*np.array([(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])
    
    # derivative w.r.t natural coordinate; d/d(xi); d/d(eta) 
    nderiv = np.zeros((4,2))
    nderiv[0,:] = -1/4*np.array([-(1-eta),-(1-xi)])
    nderiv[1,:] = -1/4*np.array([1-eta,-(1+xi)])
    nderiv[2,:] = -1/4*np.array([1+eta,1+xi])
    nderiv[3,:] = -1/4*np.array([-(1+eta),1-xi])
    return shape, nderiv