# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 23:27:45 2021

@author: Mahmu
"""

import numpy as np

amount = 10000
e1 = np.random.uniform(0,1,amount)
e2 = np.random.uniform(0,1,amount)

eList = []

for i in range(amount):
    e_i = min(e1[i],e2[i])
    eList.append(e_i)

e = np.mean(eList)

print("Expected value:", round(e, 2))

ans = [0, 0.1, 0.25, 0.4, 0.5]
abs_ans = np.abs([ans]-e)
i = np.argmin(abs_ans)
print("Expected value is closest to:", ans[i])

print("Solution is alternative [d]")