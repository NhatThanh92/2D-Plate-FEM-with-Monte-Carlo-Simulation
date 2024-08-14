# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:54:59 2024

@author: thanh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def PlotMesh(xy,eNodes,nel):
    
        # ex: size=np.size(eNodes) count whole array
        #row= np.size(eNodes,0) count row
        #colum = np.size(eNodes,1) count column
        #size, row, colum
        
    nnel = np.size(eNodes,1)
    X = np.zeros((nnel,nel))
    Y = np.zeros((nnel,nel))
    
    for iel in range(nel):
        for i in range(nnel):
            ndi = eNodes[iel,i] # acess into each element (iel=row, i =column)
            X[i,iel] = xy[ndi-1,0]
            Y[i,iel] = xy[ndi-1,1]
            
    plt.figure(figsize=(8,8))
    plt.axis('equal')
    plt.fill(X, Y, facecolor='none', edgecolor='purple', linewidth=1)
    plt.show