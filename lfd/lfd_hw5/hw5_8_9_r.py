# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:23:44 2021

@author: Mahmu
"""

import random
import numpy as np
import math
import pylab

def getPoints(amount, a = -1, b = 1):
    return np.random.uniform(-1,1,amount)
    
def getTF(A,B):
    k = (B[1] - A[1]) / (B[0] - A[0]) # A, B = [x,y]
    m = B[1] - k * B[0]  
    w_f = np.array([m, k, -1]) # y = kx+m
    return w_f
    
def LR(N,X,y_f):
    eta = 0.01
    w_g = np.zeros(3)       # weight vector for hypothesis g
    epoch_total = 0
    eps = []
    for t in range(10**5):
        indices = list(range(N)) # to go to the global min through diff directions
        random.shuffle(indices)
        w_old = w_g
        for index in indices:
            xn = X[index, :]                
            yn = y_f[index]
            delta_w = (-yn * xn / (1 + math.exp(yn * np.dot(w_g.T, xn))))
            w_g = w_g - eta * delta_w
            #print("t = ", t, "    diff_w = ", np.linalg.norm(w_g - w_old))
        if np.linalg.norm(w_g - w_old) < 0.01:
            break
    epoch_total += t
    return epoch_total, w_g
    
def getTestPoints(N_test, a = -1, b = 1):
    return np.array([np.ones(N_test),getPoints(N_test) , getPoints(N_test) ]).T  

def crossEntropy(N_test, y_f_test, X_test, w_g):
    E_in = 0
    for i in range(N_test):
        E_in += math.log(1 + math.exp(-y_f_test[i] * np.dot(X_test[i,:], w_g)))
    return E_in / N_test

def problem8_9():
    
    RUNS = 100
    N = 100
    N_test = 1000
    E_in_sum = 0
    epoch_total = 0
    
    for run in range(RUNS):
        w_f = getTF(getPoints(2), getPoints(2))
        X = np.transpose(np.array([np.ones(N), getPoints(N), getPoints(N)]))
        y_f = np.sign(np.dot(X, w_f)) # [1, x1, x2]*[m,k,-1] = [m+x1*k-x2]
        
        t, w_g = LR(N,X,y_f)
        epoch_total += t
        
        X_test = getTestPoints(N_test)
        y_f_test = np.sign(np.dot(X_test, w_f))                   
        E_in_sum += crossEntropy(N_test, y_f_test, X_test, w_g)
    
    E_out = E_in_sum / RUNS
    epoch_avg = epoch_total / RUNS
    
    return (E_out, epoch_avg)


E_out, epoch_avg = problem8_9()
print("The average cross entropy error E_out over 100 runs: ", E_out)
print("The average number of epochs: ", epoch_avg)