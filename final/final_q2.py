# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:44:48 2021

@author: Mahmut Osmanovic
"""

"""
• Bias and Variance
2. Recall that the average hypothesis ¯g was based on training the same model H
on different data sets D to get g
(D) ∈ H, and taking the expected value of g
(D)
w.r.t. D to get ¯g. Which of the following models H could result in ¯g 6∈ H?
[a] A singleton H (H has one hypothesis)
[b] H is the set of all constant, real-valued hypotheses
[c] H is the linear regression model
[d] H is the logistic regression model
[e] None of the above

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('bmh')

def theta(s):
    return 1 / (1 + np.exp(-s))

def right_hand_side(x):
    return -(np.log(( 0.5*( theta(0.5*x) + theta(0.6*x) ))**(-1) -1)) / x

def main():
    x = np.arange(-1,1, 0.01) + 0.001
    y = right_hand_side(x)
    
    plt.plot(x,y)
    plt.title("Is $w_1$ a constant?")
    plt.xlabel('$x_1$')
    plt.ylabel("$w_1$ = right hand side\n(according to assumption)")
    plt.show()
    
    
main()

"""
https://github.com/homefish/edX_Learning_From_Data_2017/blob/master/homework_9_final/hw9_p1_2_3_4_5_6.ipynb

We can see that according to the plot $w_1$ is not a constant which contradicts
the definition of $w_1$ being constant. Therefore, our assumption that the
average $\bar{g}$ withIN H is wrong and 2[d] correct.

"""