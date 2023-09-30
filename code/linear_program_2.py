# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:59:29 2023

@author: admin
"""

import numpy as np
import pandas as pd


n=99

def matrix(k):
    M = 100*np.ones((n,n))

    for i in range(n):
        for j in range(n):
            if j%k==k-1:
                if i==j:
                    M[i,j]=1
                elif j-i in [g for g in range(k)]:
                    M[i,j]=1
    if 99%k==0:
        return M
    else:
        last_one = np.where(M[::-1,-k]==1)[0][0]
        M[-last_one:,-1] = 1
        return M

M = [matrix(k) for k in range(1,6)]


def get_coeffs():
    path = 'C:\\Users\\admin\\Dropbox\\My PC (DESKTOP-UTCQBLK)\\Desktop\\'
    file = 'linear_data.xlsx'
    
    coeffs = np.array(pd.read_excel(path+file,header=None)).T
    return coeffs

coeffs = get_coeffs()


def fill_coeffs(m):
    count = -1
    for i in range(len(m.T)):
        for j in range(len(m)):
            if m.T[i,j] ==1:
                count+=1
                m[j,i] = coeffs[0][count]
    return m

M = [fill_coeffs(M[i]) for i in range(len(M))]
C = M[1]

def write_variables(n):
    X = []
    for i in range(n):
        X.append([])
        for j in range(n):
            X[i].append('Subscript[x,'+str(i)+','+str(j)+']')
    
    X = np.array(X,dtype=str)
    return X

X = write_variables(n)

def dot(X,C):
    A = []
    for i in range(n):
        A.append([])
        for j in range(n):
            
            c = str(C[i,j])
            a = c+'*'+X[i,j]
            A[i].append(a)
    return np.array(A,dtype=str)

A = dot(X,C)
    

    
    
    