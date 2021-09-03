# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:54:26 2021

@author: Mahmu
"""

import numpy as np

# Format: x1,x2, label
training = np.loadtxt("C:\\uni\\AI\\lfd_hw6\\in.txt")
x1 = training[:,0]
x2 = training[:,1]
y = training[:,2]
N = training.shape[0]

test = np.loadtxt("C:\\uni\\AI\\lfd_hw6\\out.txt")

def NONLIN(N, x1,x2):
    return np.array([np.ones(N), x1, x2, x1**2, x2**2, x1*x2,
                     np.absolute(x1-x2), np.absolute(x1+x2)]).T

"NOTE: Classification Error = fraction of missclassified points"

def linReg(lambda_param):
    Z = NONLIN(N, x1, x2)
    num_col_Z = Z.shape[1]
    Z_dagger_reg = np.dot(np.linalg.inv(np.dot(Z.T, Z) +
                      lambda_param * np.identity(num_col_Z)), Z.T)
    w = np.matmul(Z_dagger_reg,y)
    return Z, num_col_Z, w

def runQ2_inSample(lambda_param):
    Z, num_col_Z, w = linReg(lambda_param) # Use linReg to get weightvector
    error_in = sum(y != np.sign(np.dot(Z,w)))/N # 
    return error_in, w
 
error_in, w = runQ2_inSample(10**(-3))
print("The E_in is:", round(error_in,2))

def runQ2_outSample():
    N = test.shape[0]
    y = test[:,2]
    Z = NONLIN(N, test[:,0],test[:,1])
    error_out = sum(y != np.sign(np.dot(Z,w)))/N # sign(Zw) = y
    return error_out
    
error_out = round(runQ2_outSample(),2)
print("The E_out is:", error_out)

print("ANS: [d]")
