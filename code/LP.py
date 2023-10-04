# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 17:02:35 2023

@author: admin
"""

import numpy as np
import pandas as pd

def get_coeffs():
    path = 'D:\\0_Work\\'
    file = 'Book1.xlsx'

    coeffs = pd.read_excel(path+file,header=None)
    coeffs = np.concatenate([coeffs[3*i][:,None] for i in range(4)],axis=1)
    
    return coeffs.T

C = get_coeffs()

def get_variables(C):
    X = []
    for k in range(len(C)):
        X.append([])
        for j in range(len(C.T)):
            X[k].append('x_'+str(k+1)+'_'+str(j+1))
    return np.array(X)

X = get_variables(C)


def con1(X,C):
    R = [np.zeros_like(C) for j in range(len(C.T))]
    
    for j in range(len(C.T)):
        R[j][:,j] = 1
        
    return R

R1 = con1(X,C)#sum of each column equals 1

def con2(X,C):
    R = np.ones_like(C)
    for k in range(len(R)):
        R[k] = 25*2**(k)
        
    return R

R2 = con2(X,C)#sum of all equals the freq range

def con3(X,C):
    
    
    