# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:30:55 2021

@author: Mahmut Osmanovic
"""

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# https://work.caltech.edu/homework/hw8.pdf

# K(x_n, x_m) = exp(-norm(2, x_n-x_m)^2) = RBF
# Consider RBF above in soft-margin SVM approach.
# Focus on 1 vs. 5 classifier.

# Q1:
    # Which of the following values of C results
    # in the lowest E_in?
        # a) C = 0.01
        # b) C = 1
        # c) C = 100
        # d) C = 10**(4)
        # e) C = 10**(6)
        
# Q2:
    # Which of the following values of C results
    # in the lowest E_out?
        # a) C = 0.01
        # b) C = 1
        # c) C = 100
        # d) C = 10**(4)
        # e) C = 10**(6)
        
# import libraries
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.style as style
style.use('bmh')

df_train = pd.read_csv('D:\\AI\\lfd\\lfd_hw8\\features_train.txt', names =
           ['digit', 'intensity', 'symmetry'], sep='\s+', dtype=np.float64)
df_test = pd.read_csv('D:\\AI\\lfd\\lfd_hw8\\features_test.txt', names =
           ['digit', 'intensity', 'symmetry'], sep='\s+', dtype=np.float64)

ones = df_train[df_train['digit'] == 1].assign(y = np.ones(df_train[df_train['digit'] == 1].shape[0]))
fives = df_train[df_train['digit'] == 5].assign(y = -np.ones(df_train[df_train['digit'] == 5].shape[0]))
ones_test = df_test[df_test['digit'] == 1].assign(y = np.ones(df_test[df_test['digit'] == 1].shape[0]))
fives_test = df_test[df_test['digit'] == 5].assign(y = -np.ones(df_test[df_test['digit'] == 5].shape[0]))

df_1_vs_5 = ones.append(fives, ignore_index=True)
df_test_1_vs_5 = ones_test.append(fives_test, ignore_index=True)

X_train_1_vs_5 = np.c_[df_1_vs_5['intensity'], df_1_vs_5['symmetry']]
y_train_1_vs_5 = np.array(df_1_vs_5['y'])
X_test_1_vs_5 = np.c_[df_test_1_vs_5['intensity'], df_test_1_vs_5['symmetry']]
y_test_1_vs_5 = np.array(df_test_1_vs_5['y'])

fig11 = plt.figure(11, dpi = 80)
C_values = [10**k for k in [-2, 0, 2, 4, 6]]
N_train = X_train_1_vs_5.shape[0]
E_in_values = []

print("RBF classifier, 1 vs 5:")
for C in C_values:
    clf_1_vs_5_rbf = svm.SVC(C, kernel = 'rbf', gamma = 1)
    clf_1_vs_5_rbf.fit(X_train_1_vs_5,  y_train_1_vs_5)

    y_predict_train_1_vs_5_rbf = clf_1_vs_5_rbf.predict(X_train_1_vs_5)
    E_in = sum(y_train_1_vs_5 != y_predict_train_1_vs_5_rbf) / N_train

    print("C = {} => \t\tE_in = {}".format(C, E_in))
    E_in_values.append(E_in)
print()
    
    
#plt.plot(C_values, E_in_values, 'bo-', label='RBF Kernel')
plt.semilogx(C_values, E_in_values, 'bo-', label='RBF Kernel')

plt.xlabel('C')
plt.ylabel('E_in')
plt.title('Problem 9 - RBF Kernel')
plt.legend()
plt.show()

"""
Result for Problem 9
The lowest $E_{in}$ is achieved for the highest value of $C = 10^6$ which
is to be expected if you recall that $C$ is the penalty factor. For 
$C = \infty$ we had the hard margin case with zero in-sample error. 
The correct answer is therefore 9[e].
"""

fig12 = plt.figure(12, dpi = 80)
C_values = [10**k for k in [-2, 0, 2, 4, 6]]
N_test = X_test_1_vs_5.shape[0]
E_out_values = []

print("RBF classifier, 1 vs 5:")
for C in C_values:
    clf_1_vs_5_rbf = svm.SVC(C, kernel = 'rbf', gamma = 1)
    clf_1_vs_5_rbf.fit(X_train_1_vs_5,  y_train_1_vs_5)

    y_predict_test_1_vs_5_rbf = clf_1_vs_5_rbf.predict(X_test_1_vs_5)
    E_out = sum(y_test_1_vs_5 != y_predict_test_1_vs_5_rbf) / N_test

    print("C = {} => \t\tE_out = {}".format(C, E_out))
    E_out_values.append(E_out)
print()
    
    
plt.semilogx(C_values, E_out_values, 'bo-', label='RBF Kernel')

plt.xlabel('C')
plt.ylabel('E_out')
plt.title('Problem 10 - RBF Kernel')
plt.legend()
plt.show()

"""
Result for Problem 10
The lowest E_{out} is achieved for C = 100, thus answer 10[c] is correct.
"""