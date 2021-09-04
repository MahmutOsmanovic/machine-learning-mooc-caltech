# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 09:28:50 2021

@author: Mahmu

Regularization with Weight Decay
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

def linReg():
    Z = NONLIN(N, x1, x2)
    Z_dagger = np.dot(np.linalg.inv(np.dot(Z.T, Z)), Z.T)
    w = np.matmul(Z_dagger,y)
    return Z, w

def runQ2_inSample():
    Z, w = linReg() # Use linReg to get weightvector
    error_in = sum(y != np.sign(np.dot(Z,w)))/N # 
    return error_in, w
 
error_in, w = runQ2_inSample()
print("The E_in is:", round(error_in,2))

def runQ2_outSample():
    N = test.shape[0]
    y = test[:,2]
    Z = NONLIN(N, test[:,0],test[:,1])
    error_out = sum(y != np.sign(np.dot(Z,w)))/N # sign(Zw) = y
    return error_out
    
error_out = round(runQ2_outSample(),2)
print("The E_out is:", error_out)

print("ANS: [a]")




