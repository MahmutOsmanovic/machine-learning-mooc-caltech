# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 07:01:08 2021

@author: Mahmu
"""

# HYPOTHESIS: h(x) = ax^2 + b

import numpy as np


def problem4():
    
    RUNS = 1000000
    a_total = 0
    b_total = 0
    N = 2          # size of data set
    
    for _ in range(RUNS):
        # two random points
        x_rnd = np.random.uniform(-1, 1, N)
        y_rnd = np.sin(np.pi * x_rnd)

        # linear regression for model y = ax^2
        X = np.array([np.ones(N), x_rnd * x_rnd]).T
        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y_rnd)
        b, a = w

        a_total += a
        b_total += b
        
    a_avg = a_total / RUNS
    b_avg = b_total / RUNS
    return b_avg, a_avg

print("h(x) = b + ax^2")
print("solution problem 7: (b_avg, a_avg) = ", problem4())


#-------------------------------------------------------------------------


def problem5():
    N_test = 1000
    x_test = np.random.uniform(-1,1,N_test)

    y_f = np.sin(np.pi * x_test)
    b_avg, a_avg = problem4()
    y_g_bar = b_avg + a_avg * (x_test * x_test)

    bias = sum((y_f - y_g_bar)**2) / N_test
    return bias
    

print("\nSolution to problem 7: bias = ", problem5())

#--------------------------------------------------------------------------

def problem6():
    b_avg, a_avg = problem4()
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
            x_rnd = np.random.uniform(-1, 1, N)
            y_rnd = np.sin(np.pi * x_rnd)

            # linear regression for model y = ax^2
            # get a particular g^(D)
            X = np.array([np.ones(N), x_rnd * x_rnd]).T
            w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y_rnd)
            b, a  = w
            
            # calculate difference on test point
            y_g = b + a * x_test**2
            y_g_bar = b_avg + a_avg * x_test**2
            expectation_over_D += (y_g - y_g_bar)**2 / RUNS_D

        expectation_over_X += expectation_over_D / RUNS_X
    
    variance = expectation_over_X
    return variance


print("\nSolution to problem 7, variance = ", problem6())