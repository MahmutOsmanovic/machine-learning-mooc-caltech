# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pylab

"""
We are given 10 input units, 1 output unit and 36 units in the hidden layer 
(a unit also includes the bias node). We have to find an architecture for the 
shidden layer that minimizes the number of weight edges.

The minimum number of weight edges is achieved for the architecture with 18 hidden 
layers, where each layer consists of a bias node and a regular node:
    d = [9,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
"""

def get_edges(architecture):
    total_edges = 0
    L = len(architecture) - 1
    for i in range(L):
        total_edges += (d[i] + 1) * d[i+1]
    return total_edges

d = [9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print("total number of edges: ", get_edges(d))