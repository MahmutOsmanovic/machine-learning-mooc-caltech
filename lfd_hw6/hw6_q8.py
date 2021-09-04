# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pylab

"""
A fully connected Neural Network has "2" layers; layer 0, layer 1 and layer 2.
Features; input-layer = 5, hidden layer = 3, output layer = 1

- What is the closest amout of iterations in a single iteration of backpropogation 
(using SGD on one data point)?
"""

# STEP 0: Forward propagation
"""
The input is a data point (X,y).
Calculate the signals: s_(l) = (W_(l))X_(l-1) and save the outputs:
    X_(l) = activiation-function(s_(l)) for layers l = 1,2,...,L.
    
1
*  1
*  *   
*  *  *
*  * 
*

6 + 6 + 6 = 18, s_(1)

3 + 1 = 4 = s_(2)

TOTAL: 22 iteration for forward propogation =/= #weight edges
"""

# STEP 1: Compute the sensitivities lamba_(l)

"""
lambda_(l) = 2(X_(l) - y)(t'(s_(L))); L = 2 => lambda_(2) = 2(X_(2) - y)(t'(s_(2)))
Choose an activation function t whose derivative is t'(s) = (1-t(s)^(2));
Choose: t(s) = tanh(s) => t'(s) = (1-t(s)^(2)) = (1-x^2),
lambda_(2) =  2(X_(2) - y)(1-x_(2)^2)

...

=> 3.
"""

# STEP 2: Compute the partial derivatives de/dW_(l)

"""
6*3 + 4*1 = 22
"""

# TOTAL = 22 + 3 + 22 = 47 => ANSWER = 47 => ANSWER = 8[d].

def N_its(d):
    
    total_its = 0
    L = len(d) - 1
    
    for i in range(1, L+1):
        total_its += ((d[i-1] + 1) * d[i])
    total_its *= 2
        
    for i in range(1, L):
        total_its += d[i]
    
    return total_its

d = [5,3,1]
print(N_its(d))