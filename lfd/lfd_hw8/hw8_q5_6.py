# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 12:33:17 2021

@author: Mahmu
"""

# import libraries
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.style as style
style.use('bmh')

# Q5
df_train = pd.read_csv('D:\\AI\\lfd\\lfd_hw8\\features_train.txt', names = 
                       ['digit', 'intensity', 'symmetry'], sep='\s+', dtype=np.float64)
"""print(df_train.head(5), end='\n\n')
print(df_train.describe(), end='\n\n')
print(df_train.shape)"""

# Consider 1 vs 5 classifier
# We choose only rows with digits == 1 or digits == 5
df_train[df_train['digit'] == 1]
df_train[df_train['digit'] == 5]

# Append a column y with labels y=+1 for digit==1
ones = df_train[df_train['digit'] == 1].assign(y = np.ones(df_train[df_train['digit'] == 1].shape[0]))
print("Number of rows and columns for 'ones' dataframe: ", ones.shape)
#print(ones.head(5), end='\n\n')

# Append a column y with labels y=-1 for digit==5
# https://chrisalbon.com/python/pandas_assign_new_column_dataframe.html
fives = df_train[df_train['digit'] == 5].assign(y = -np.ones(df_train[df_train['digit'] == 5].shape[0]))
print("Number of rows and columns for 'fives' dataframe: ", fives.shape)
#print(fives.head(5), end='\n\n')

# Glue together the dataframes for ones and fives
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.append.html
df_1_vs_5 = ones.append(fives, ignore_index=True)
print("Number of rows and columns of 1-vs-5 dataframe: ", df_1_vs_5.shape, end='\n\n')

print("- First five rows of 1-vs-5 dataframe:\n")
print(df_1_vs_5.head(5), end='\n\n')

print("- Last five rows of 1-vs-5 dataframe:\n")
print(df_1_vs_5.tail(5), end='\n\n')

print("- Statistical information about 1-vs-5 dataframe:\n")
print(df_1_vs_5.describe(), end='\n\n')

plt.hist(df_1_vs_5['digit'], edgecolor = 'black')
plt.title('Distribution of digits 1 and 5 in the 1-vs-5 training set')
plt.xlabel('digit')
plt.ylabel('count')
plt.show()

# Training data
X_train_1_vs_5 = np.c_[df_1_vs_5['intensity'], df_1_vs_5['symmetry']]
print(X_train_1_vs_5.shape)

# labels
y_train_1_vs_5 = np.array(df_1_vs_5['y'])
print(y_train_1_vs_5.shape)
# Note: I'm printing the shapes of the matrices to check if I have extracted the data correctly.

# explore Option 5[a] and 5[b]

fig5 = plt.figure(5, dpi = 80)
C_values = []
num_support_values = []

for C in [0.001, 0.01, 0.1, 1]:
    clf_1_vs_5 = svm.SVC(C, kernel = 'poly', degree = 2, coef0 = 1, gamma = 1)
    clf_1_vs_5.fit(X_train_1_vs_5,  y_train_1_vs_5)
    print("C = {} => \t\t num support vectors = {}".format(C, sum(clf_1_vs_5.n_support_)))
    C_values.append(C)
    num_support_values.append(sum(clf_1_vs_5.n_support_))
    
    
#plt.plot(C_values, num_support_values, 'bo-')
plt.semilogx(C_values, num_support_values, 'bo-')

plt.xlabel('C')
plt.ylabel('number of support vectors')
plt.title('Exploring answers 5[a] and 5[b]')
plt.show()

# Result: This excludes option 5[a] and 5[b].

# Explore Option 5[c]
# E_out goes down when C goes up ???

# Let's first read in the test set
df_test = pd.read_csv('D:\\AI\\lfd\\lfd_hw8\\features_test.txt', names = ['digit', 'intensity', 'symmetry'], sep='\s+', dtype=np.float64)

print("- Printing the first five rows of the test set:\n")
print(df_test.head(5))

print("\n\n- Some statistical information about the test set:\n")
print(df_test.describe())

print("\n\n- Number of rows and columns in the test set table:", df_test.shape)

print("\n\n- Number of rows with digit 1:", df_test[df_test['digit'] == 1].shape[0])
print("- Number of rows with digit 5:", df_test[df_test['digit'] == 5].shape[0])

# Consider test set for 1-vs-5 classifier

# Append a column y with labels y=+1 for digit==1
ones_test = df_test[df_test['digit'] == 1].assign(y = np.ones(df_test[df_test['digit'] == 1].shape[0]))
print("Number of rows and columns for 'ones_test' dataframe: ", ones_test.shape)

# Append a column y with labels y=-1 for digit==5
# https://chrisalbon.com/python/pandas_assign_new_column_dataframe.html
fives_test = df_test[df_test['digit'] == 5].assign(y = -np.ones(df_test[df_test['digit'] == 5].shape[0]))
print("Number of rows and columns for 'fives_test' dataframe: ", fives_test.shape)

# Glue together the dataframes ones_test and fives_test
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.append.html
df_test_1_vs_5 = ones_test.append(fives_test, ignore_index=True)

print("Number of rows and columns in the dataframe", df_test_1_vs_5.shape)

print("\n\n- First five rows of 1_vs_5_test dataframe:\n")
print(df_test_1_vs_5.head(5), end='\n\n')

print("- Last five rows of 1_vs_5_test dataframe:\n")
print(df_test_1_vs_5.tail(5), end='\n\n')

print("- Statistical information about 1_vs_5_test dataframe:\n")
print(df_test_1_vs_5.describe(), end='\n\n')


plt.hist(df_test_1_vs_5['digit'])
plt.title("Distribution of digits 1 and 5 in the 1-vs-5 test set")
plt.xlabel('digit')
plt.ylabel('count')
plt.show()

# Create X_test
X_test_1_vs_5 = np.c_[df_test_1_vs_5['intensity'], df_test_1_vs_5['symmetry']]
print("Number of rows and columns in X_test_1_vs_5: ", X_test_1_vs_5.shape)

y_test_1_vs_5 = np.array(df_test_1_vs_5['y'])
print("Number of rows and columns in y_test_1_vs_5: ", y_test_1_vs_5.shape)

# explore Option 5[c]

fig6 = plt.figure(6, dpi = 80)
C_values = []
E_out_values = []
N_test = X_test_1_vs_5.shape[0]

for C in [0.001, 0.01, 0.1, 1]:
    
    # setting up the classifier
    clf_1_vs_5 = svm.SVC(C, kernel = 'poly', degree = 2, coef0 = 1, gamma = 1) # choose model
    clf_1_vs_5.fit(X_train_1_vs_5,  y_train_1_vs_5) # train with that model to obtain optimal weights
    
    # predictions made by classifier
    y_predict_test_1_vs_5 = clf_1_vs_5.predict(X_test_1_vs_5) # use these weights to make predictions on test dataset
    
    # E_out is the fraction of mismatches on the test set
    E_out = sum(y_test_1_vs_5 != y_predict_test_1_vs_5) / N_test
    
    print("C = {} => \t\tE_out = {}".format(C, E_out))
    C_values.append(C)
    E_out_values.append(E_out)
    
    
# Use logarithmic scale on x-axis
plt.semilogx(C_values, E_out_values, 'bo-')

plt.xlabel('C')
plt.ylabel('E_out')
plt.title('Exploring multiple choice answer 5[c]')
plt.show()

# Does the maximum C achieve the lowest E_in ?
# explore Option 5[d]

fig7 = plt.figure(7, dpi = 80)
C_values = []
E_in_values = []
N_train = X_train_1_vs_5.shape[0]

for C in [0.001, 0.01, 0.1, 1]:
    clf_1_vs_5 = svm.SVC(C, kernel = 'poly', degree = 2, coef0 = 1, gamma = 1)
    clf_1_vs_5.fit(X_train_1_vs_5,  y_train_1_vs_5)
    
    y_predict_train_1_vs_5 = clf_1_vs_5.predict(X_train_1_vs_5)
    E_in = sum(y_train_1_vs_5 != y_predict_train_1_vs_5) / N_train
    
    print("C = {} => \t\tE_in = {}".format(C, E_in))
    C_values.append(C)
    E_in_values.append(E_in)
    
    
plt.semilogx(C_values, E_in_values, 'bo-')

plt.xlabel('C')
plt.ylabel('E_in')
plt.title('Explore option 5[d]')
plt.show()

# The maximum C achieves the lowest E_{in}, thus the answer is 5[d]

# train two classifiers , Q = 2 and Q = 5 respectively
# plot E_in vs C

# explore Option 5[c]

fig8 = plt.figure(8, dpi = 80)
C_values = [0.0001, 0.001, 0.01, 0.1, 1]
N_train = X_train_1_vs_5.shape[0]

E_in_values = [[],[]]  # E_in_values for Q = 2 and Q = 5 respectively
Q_values = [2, 5]


for i in range(2):
    for C in C_values:
        
        # sweep values for C and Q
        clf_1_vs_5 = svm.SVC(C, kernel = 'poly', degree = Q_values[i], coef0 = 1, gamma = 1)
        clf_1_vs_5.fit(X_train_1_vs_5,  y_train_1_vs_5)

        y_predict_train_1_vs_5 = clf_1_vs_5.predict(X_train_1_vs_5)
        E_in = sum(y_train_1_vs_5 != y_predict_train_1_vs_5) / N_train

        print("Q = {}, C = {} => \t\tE_in = {}".format(Q_values[i], C, E_in))
        E_in_values[i].append(E_in)
    print()
    
    
plt.semilogx(C_values, E_in_values[0], 'bo-', label='Q=2')
plt.semilogx(C_values, E_in_values[1], 'ro-', label='Q=5')


plt.xlabel('C')
plt.ylabel('E_in')
plt.title('Explore options 6[a] and 6[c]')
plt.legend()
plt.show()

# QUESTION 6
# train two classifiers , Q = 2 and Q = 5 respectively
# plot E_in vs C

# explore Option 5[c]

fig8 = plt.figure(8, dpi = 80)
C_values = [0.0001, 0.001, 0.01, 0.1, 1]
N_train = X_train_1_vs_5.shape[0]

E_in_values = [[],[]]  # E_in_values for Q = 2 and Q = 5 respectively
Q_values = [2, 5]


for i in range(2):
    for C in C_values:
        
        # sweep values for C and Q
        clf_1_vs_5 = svm.SVC(C, kernel = 'poly', degree = Q_values[i], coef0 = 1, gamma = 1)
        clf_1_vs_5.fit(X_train_1_vs_5,  y_train_1_vs_5)

        y_predict_train_1_vs_5 = clf_1_vs_5.predict(X_train_1_vs_5)
        E_in = sum(y_train_1_vs_5 != y_predict_train_1_vs_5) / N_train

        print("Q = {}, C = {} => \t\tE_in = {}".format(Q_values[i], C, E_in))
        E_in_values[i].append(E_in)
    print()
    
    
plt.semilogx(C_values, E_in_values[0], 'bo-', label='Q=2')
plt.semilogx(C_values, E_in_values[1], 'ro-', label='Q=5')


plt.xlabel('C')
plt.ylabel('E_in')
plt.title('Explore options 6[a] and 6[c]')
plt.legend()
plt.show()

# train two classifiers , Q = 2 and Q = 5 respectively
# plot E_out vs C

# explore Option 6[d]

fig9 = plt.figure(9, dpi = 80)
C_values = [0.0001, 0.001, 0.01, 0.1, 1]
N_test = X_test_1_vs_5.shape[0]

E_out_values = [[],[]]  # E_in_values for Q = 2 and Q = 5 respectively
Q_values = [2, 5]

for i in range(2):
    for C in C_values:
        
        # train using the training set!
        clf_1_vs_5 = svm.SVC(C, kernel = 'poly', degree = Q_values[i], coef0 = 1, gamma = 1)
        clf_1_vs_5.fit(X_train_1_vs_5,  y_train_1_vs_5)

        # predict using the test set!
        y_predict_test_1_vs_5 = clf_1_vs_5.predict(X_test_1_vs_5)
        E_out = sum(y_test_1_vs_5 != y_predict_test_1_vs_5) / N_test

        print("Q = {}, C = {} => \t\tE_out = {}".format(Q_values[i], C, E_out))
        E_out_values[i].append(E_out)
    print()
    

plt.semilogx(C_values, E_out_values[0], 'bo-', label='Q=2')
plt.semilogx(C_values, E_out_values[1], 'ro-', label='Q=5')


plt.xlabel('C')
plt.ylabel('E_out')
plt.title('Option 6[a]')
plt.legend()
plt.show()

# train two classifiers , Q = 2 and Q = 5 respectively
# plot number of support vectors

# explore Option 6[d]

fig10 = plt.figure(10, dpi = 80)
C_values = [0.0001, 0.001, 0.01, 0.1, 1]
N_train = X_train_1_vs_5.shape[0]

num_support_vectors_values = [[],[]]  # Number of Support vectors for Q = 2 and Q = 5 respectively
Q_values = [2, 5]

for i in range(2):
    for C in C_values:
        # train using the training set!
        clf_1_vs_5 = svm.SVC(C, kernel = 'poly', degree = Q_values[i], coef0 = 1, gamma = 1)
        clf_1_vs_5.fit(X_train_1_vs_5,  y_train_1_vs_5)
        print("HIIII IAM THE SUPP VECTORS :D", clf_1_vs_5.n_support_)
        num_support_vectors = sum(clf_1_vs_5.n_support_)
        
        print("Q = {}, C = {} => \t\tnum_support_vectors = {}".format(Q_values[i], C, num_support_vectors))
        num_support_vectors_values[i].append(num_support_vectors)
    print()
    
    

plt.semilogx(C_values, num_support_vectors_values[0], 'bo-', label='Q=2')
plt.semilogx(C_values, num_support_vectors_values[1], 'ro-', label='Q=5')



plt.xlabel('C')
plt.ylabel('number of support vectors')
plt.title('Exploring option 6[b]')
plt.legend()
plt.show()


# The number of support vectors at C = 0.001 is lower for Q = 5,
# therefore answer 6[b] is correct. Read graph.

# -----------------------------------------------------------
# The degree parameter (Q) controls the flexibility of the 
# decision boundary. Higher degree kernels yield a more flexible
# decision boundary
# https://stats.stackexchange.com/questions/348318/degree-parameter-for-svm-polynomial-kernel
# -----------------------------------------------------------
# Intuitively, the gamma parameter defines how far the influence
# of a single training example reaches, with low values meaning
# ‘far’ and high values meaning ‘close’. The gamma parameters can
# be seen as the inverse of the radius of influence of samples
# selected by the model as support vectors.
# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
# -----------------------------------------------------------
# C parameter trades off correct classification of training 
# examples against maximization of the decision function’s margin.
# For larger values of C, a smaller margin will be accepted if 
# the decision function is better at classifying all training 
# points correctly. A lower C will encourage a larger margin, 
# therefore a simpler decision function, at the cost of training
# accuracy. In other words C behaves as a regularization
# parameter in the SVM.
# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
# -----------------------------------------------------------

































