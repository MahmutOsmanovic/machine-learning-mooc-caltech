# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:00:45 2021

@author: Mahmu
"""

import numpy as np
from matplotlib import image
from matplotlib import pyplot

# https://work.caltech.edu/homework/hw3.pdf

def getN(ProbabilityBound, epsilon, M):
    return print(str(-np.log(ProbabilityBound/(2*M))/(2*(epsilon**2))))


# 1. 
#getN(0.03,0.05,1)

# 2.
#getN(0.03,0.05,10)

# 3.
#getN(0.03,0.05,100)

# 4.
"""4 points works. 5 points is the breakpoint. 4 of them will make a tetrahedron, on left out. 
   Call the utmost left point within the tetrahedron x2 and the point outside of it x4. X4 is
   placed at the utmost right point of the tetrahedron. Atleast one point will always be placed 
   outside of it. It is not possible to separate x2 and x4 as red points from x5,x1,x3 as blue.
   => Breakpoint = 5
"""

# 5.
""" i: Since: (N 1) = N and (N 0) = 1 => N + 1, OK
    ii: (N 2) + (N 1) + (N 0) => OK
        How did you classify a polynomial from not a polynomial?
        Answer: An algebraic expression is not a polynomial when there are square roots, negative powers,
                and variables in the denominator of any fractions.
    iii: No. Results in a non-polynomial function of N.
    iv:  No. Is trivially not a polynoial function of N, which a growth function by necessity must be.
    v: OK
"""

# 6.
"""
https://rpubs.com/giuliano_mega/376525
"""

#7
"""
https://rpubs.com/giuliano_mega/376525
"""

#8
"""
https://rpubs.com/giuliano_mega/376525
"""

#9
"""
https://rpubs.com/giuliano_mega/376525
"""

#10
"""
https://rpubs.com/giuliano_mega/376525
"""