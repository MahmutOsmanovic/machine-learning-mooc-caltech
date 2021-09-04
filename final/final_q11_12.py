# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:53:37 2021

@author: Mahmut Osmanovic
"""

from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.style as style
style.use('bmh')

# Q11
# Write down data
data = [[1, 0, -1],
        [0, 1, -1],
        [0, -1, -1],
        [-1, 0, 1],
        [0, 2, 1],
        [0, -2, 1],
        [-2, 0, 1]]

df = pd.DataFrame(data, columns = ['x1', 'x2', 'y'], dtype=np.float64)

# transform to Z-space

x1 = df['x1']
x2 = df['x2']
y = df['y']
N = x1.shape[0]

z1 = x2*x2 - 2*x1 - 1
z2 = x1*x1 - 2*x2 + 1
N = z1.shape[0]
Z = np.c_[z1, z2]
print("points in transformed Z-space:")
print(Z)

# Target distribution: points (z1, z2, y)
plt.plot(z1[y==1], z2[y==1], 'ro', label='$y=+1$')
plt.plot(z1[y==-1], z2[y==-1], 'bo', label='$y=-1$')
plt.title("Target distribution in Z-space")
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.legend()
plt.show()

"""
We see that if there was a sep-plane at x = 0.5, it would sperate the points.
Its normal vector is n=(1,0). To find b in w^(T)z+b=0, we pick a point on the
plane, which is defined by Z, i.e. say point, p=(0.5,0). Now, the weights, for
example w=(1,0) obtain b = -0.5.
The support vectors are (0,-1), (0,3) and (1,2), equally distanced from the
sep-plane.

"""

# Q12
"""
We apply the hard-margin (meaning $C = \infty$) SVM algorithm with the kernel 
$K(x, x') = (1 + x^T x')^2$. We have to compute the number of support vectors.
"""
# classifier clf with the parameters as stated by the homework problem
clf = svm.SVC(C = np.inf, kernel = 'poly', degree = 2, coef0 = 1, gamma = 1)
clf.fit(Z, y)
print(Z.shape)
print(y.shape)

print("number of support vectors: ", sum(clf.n_support_))
# ANSWER = 12[c]