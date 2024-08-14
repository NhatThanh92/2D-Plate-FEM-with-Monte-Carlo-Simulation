# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:56:14 2024

@author: thanh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Shapefunc import shapeFuncQ4
from Jacob import Jacobian
from GaussQuad import gaussQuadrature
# nDof: num degree of freedom, nE: num of element; 
# eNodes : element connectivity
# nP = nDof/2


def formStiffness2D(nDof,nE,eNodes,nP,xy,C,h):
    K = np.zeros((nDof,nDof))
    gaussWt, gaussLoc = gaussQuadrature(1)
    for e in range(nE):
        
        # INDEX : 2D array access into 1st row and 2nd column :  a[0,1] 
        
        id  = eNodes[e,:]
        eDof = np.zeros((8,1))
        eDof[0:4,0] = id      # (u) 
        eDof[4:8,0] = id + nP # (v)
        eDof = eDof.flatten() # flatten converting a nested data structure into 1-D structure
        
        ndof = id.size
        # loop for Gauss point
        for q in range(gaussWt.size):
            
            GaussPoint = gaussLoc[q,:]
            xi = GaussPoint[0]
            eta = GaussPoint[1]
            
            #shape functions and derivatives
            shape,nDeriv = shapeFuncQ4(xi,eta)
            
            #Jacobian matrix, inverse of Jacobian
            J, xyDeriv = Jacobian(xy[id-1,:],nDeriv) #(python due to index start from 0 :id-1)
            
            # B matrix (Linear strain - displacement matrix)
            B = np.zeros((3,2*ndof)) # B(3x8)
            B[0,0:ndof] = np.transpose(xyDeriv[:,0])
            B[1,ndof:(2*ndof)]= np.transpose(xyDeriv[:,1])
            B[2,0:ndof] = np.transpose(xyDeriv[:,1])
            B[2,ndof:(2*ndof)] = np.transpose(xyDeriv[:,0])
            
            # Stiffness matrix
            BT = np.transpose(B)
            detJ = np.linalg.det(J)
            Ke = np.matmul(np.matmul(BT,C),B)*h*detJ*gaussWt[q]
            
            # Global stiffness matrix
            # K-row
            for ii in range(np.size(Ke,0)):
                row = int(eDof[ii])-1 # eDof: float, -1 dueto python index start from  0
                for jj in range(np.size(Ke,1)):
                    col = int(eDof[jj])-1
                    K[row,col] = K[row,col] + Ke[ii,jj]
    return K






