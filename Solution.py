# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:52:08 2024

@author: thanh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def solution(nDof,fixDof,K,force):
    # np.setdiff1d: eliminate fixDof, 
    #np.arange(nDof): create array with num Dof
    
    activeDof = np.setdiff1d(np.arange(nDof), fixDof)
    #******
    # np.linalg.solve= Solve a linear matrix equation, 
    # numpy.ix_(*args): Construct an open mesh from multiple sequences
    # ex: ixgrid = np.ix_([0, 1], [2, 4])
    # => This creates an index grid using np.ix_(). It selects rows 0 and 1 and columns 2 and 4 from the array a.
    #*******
    U = np.linalg.solve(K[np.ix_(activeDof,activeDof)],force[activeDof])
    
    # disp of the whole system (with fixDof)
    
    disp = np.zeros((nDof,1))
    # Assigning the matrix U to the element of disp at the index 
    # represented by activeDof
    
    disp[activeDof] = U  
    
    return disp
