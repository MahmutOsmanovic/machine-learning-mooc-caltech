# -*- coding: utf-8 -*-
"""
Created on: Wed Jul  7 15:30:17 2021

@author: Mahmut Osmanovic
"""

import matplotlib.pyplot as plt

S = 1
P = 6

S_B_P = 0

its = 5
while its:
    print(S, P)
    S_B_P += (S > P)
    S += 1
    P -= 1

    its -= 1
    
print("ANS =", S_B_P)

plt.plot([-1, 1], [-1,1])