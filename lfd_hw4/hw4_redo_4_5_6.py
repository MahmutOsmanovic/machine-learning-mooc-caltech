# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 05:07:33 2021

@author: Mahmu
"""

import numpy as np

def f(x):
    return np.sin(np.pi*x)

def problem4():
    
    RUNS = 1000 
    a_total = 0
    N = 2 # size of data set
    
    for _ in range(RUNS):
        # two random points
        x_rnd = np.random.uniform(-1,1,N) # this is a vector of size 2
        y_rnd = f(x_rnd)
        
        # linear regression for model y = ax
        X_a = np.array([x_rnd]).T
        w_a = np.linalg.solve(X_a.T @ X_a, X_a.T @ y_rnd)
        a = w_a[0]
        
        a_total += a
        
    a_avg = a_total / RUNS
    return a_avg

print("Solution problem 4: a_avg = ", problem4())
print("Answer 4[e] is therefore correct.")

#-------------------------------------------------------------------------

def problem5():
    N_test = 1000
    x_test = np.random.uniform(-1,1, N_test)
    
    y_f = f(x_test)
    a_avg = problem4()
    y_g_bar = a_avg * x_test
    
    bias = sum((y_f - y_g_bar)**2) / N_test
    return bias 

print("\nSolution to problem 5: bias = ", problem5())
print("Answer 5[b] is therefore correct.")

#-------------------------------------------------------------------------

def problem6():
    a_avg = problem4()
    expectation_over_X = 0
    
    RUNS_D = 100
    RUNS_X = 1000
    # variance: Compare each g to g_bar
    
    for i in range(RUNS_X):
        N = 2
        x_test = np.random.uniform(-1,1)
        expectation_over_D = 0
        
        for _ in range(RUNS_D):
            # two random points as data set D
            x_rnd = np.random.uniform(-1,1,N)
            y_rnd = f(x_rnd)
            
            # linear regression for model y = ax
            # get a particular g^(D)
            X_a = np.array([x_rnd]).T
            w_a = np.linalg.solve(X_a.T @ X_a, X_a.T @ y_rnd)
            a = w_a[0]
            
            # calculate difference
            y_g = a * x_test
            y_g_bar = a_avg * x_test
            expectation_over_D += (y_g - y_g_bar)**2 / RUNS_D
            
        expectation_over_X += expectation_over_D / RUNS_X
        
        variance = expectation_over_X
        return variance
        
            
            
            
            
            