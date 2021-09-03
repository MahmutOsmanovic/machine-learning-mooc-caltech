# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 21:37:40 2021

@author: Mahmut Osmanovic
"""

# Problem 7 - Regularized Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.style as style
style.use('bmh')

train = pd.read_csv("D:\\AI\\lfd\\final\\features.train.txt", names = ['digit', 'intensity', 'symmetry'], sep='\s+', dtype=np.float64)

TRAIN_DATA = np.loadtxt("D:\\AI\\lfd\\final\\features.train.txt")

# training data ignoring 0-th column which contains the digit labels
X_TRAIN = TRAIN_DATA[:,1:]

# 0-th column contains digit labels
DIGIT_LABELS_TRAIN = TRAIN_DATA[:, 0]

# number of training points
N_TRAIN = X_TRAIN.shape[0]

def get_labels(x, digit_labels):
    '''
    - Takes integer 'x' (digit 0-9)
    - Takes labels
    - Returns new labels
      where the label is 1 for digit == x, and -1 otherwise.
    '''
    y = []
    for d in digit_labels:
        if d == x:
            y.append(1)
        else:
            y.append(-1)
    return np.array(y)


# We will use regularization with weight decay (see slide 11-13 of lecture 12).

def linear_regression_reg(Z, y, lambda_param):
    '''
    - Takes feature matrix Z with rows of the form (1, z1, z2, ..., zd)
    - Takes labels y
    - Takes lambda parameter lambda_param
    - returns weight vector w_tilde_reg 
    '''    
    num_columns_Z = Z.shape[1]
    
    # see lecture 12, slide 11
    Z_dagger_reg = np.dot(np.linalg.inv(np.dot(Z.T, Z) + lambda_param * np.identity(num_columns_Z)), Z.T)

    # Use linear regression to get weight vector
    w_tilde_reg = np.dot(Z_dagger_reg, y)

    return w_tilde_reg

# Compute the in-sample error E_in

def problem7():
    
    Z_TRAIN = np.c_[np.ones(N_TRAIN), X_TRAIN]
    digit_list = [5,6,7,8,9]
    E_in_list = []
    lambda_value = 1

    min_E_in = np.inf
    min_digit = None

    for digit in digit_list:
        y = get_labels(digit, DIGIT_LABELS_TRAIN)
        w_tilde_reg = linear_regression_reg(Z_TRAIN, y, lambda_value)
        predicted_y = np.sign(np.dot(Z_TRAIN, w_tilde_reg))
        E_in = sum(predicted_y != y) / N_TRAIN
        E_in_list.append(E_in)

        if E_in < min_E_in:
            min_E_in = E_in
            min_digit = digit


    print("\nQ7\nThe lowest in-sample error: \nE_in = {:0.5f}, \nis achieved for the {}-vs-all classifier".format(min_E_in, min_digit))

    plt.plot(digit_list, E_in_list, 'ro-')
    plt.ylabel("in-sample error $E_{in}$")
    plt.xlabel("digit")
    plt.title("HW 9 - Problem 7")
    plt.show()
    

problem7()
# ANSWER = 7[d]

# Q8
# Reading in data with np.loadtxt instead of Pandas
TEST_DATA = np.loadtxt('D:\\AI\\lfd\\final\\features.test.txt')

# test data ignoring 0-th column which contains the digit labels
X_TEST = TEST_DATA[:, 1:]

# 0-th column contains digit labels
DIGIT_LABELS_TEST = TEST_DATA[:, 0]

# number of test points
N_TEST = X_TEST.shape[0]

# 1. We first have to apply the transform 
#    z = (1, x_1, x_2, x_1 x_2, x_1^2, x_2^2) for our feature matrix Z.
# 2. This time we consider the digits in [0,1,2,3,4].
# 3. We compute the out-of sample error E_{out}.

# Compute the out-of-sample error E_out
def problem8():

    x1 = X_TRAIN[:,0]
    x2 = X_TRAIN[:,1]
    Z_TRAIN = np.c_[np.ones(N_TRAIN), x1, x2, x1*x2, x1*x1, x2*x2]

    #------------

    x1 = X_TEST[:,0]
    x2 = X_TEST[:,1]
    Z_TEST = np.c_[np.ones(N_TEST), x1, x2, x1*x2, x1*x1, x2*x2]

    #------------

    digit_list_2 = [0,1,2,3,4]
    E_out_list = []
    lambda_value = 1

    #------------

    min_E_out = np.inf
    min_digit = None

    for digit in digit_list_2:

        # train with training data!
        y_train = get_labels(digit, DIGIT_LABELS_TRAIN)
        w_tilde_reg = linear_regression_reg(Z_TRAIN, y_train, lambda_value)

        # compute E_out by using test data!
        predicted_y_test = np.sign(np.dot(Z_TEST, w_tilde_reg))
        y_test = get_labels(digit, DIGIT_LABELS_TEST)
        E_out = sum(predicted_y_test != y_test) / N_TEST
        E_out_list.append(E_out)

        if E_out < min_E_out:
            min_E_out = E_out
            min_digit = digit


    print("\nQ8\nThe lowest out-of-sample error: \nE_out = {:0.5f}, \nis achieved for the {}-vs-all classifier".format(min_E_out, min_digit))

    plt.plot(digit_list_2, E_out_list, 'bo-')
    plt.ylabel("out-of-sample error $E_{out}$")
    plt.xlabel("digit")
    plt.title("HW 9 - Problem 8")
    plt.show()
    
problem8()
# ANSWER = 8[b]

# Q9
# We first compare $E_{out}$ without transform versus E_{out} with transform.

# Compute the in-sample error E_out
def problem9_E_out_without_transform():

    x1 = X_TRAIN[:,0]
    x2 = X_TRAIN[:,1]
    Z_TRAIN = np.c_[np.ones(N_TRAIN), x1, x2]

    #------------

    x1 = X_TEST[:,0]
    x2 = X_TEST[:,1]
    Z_TEST = np.c_[np.ones(N_TEST), x1, x2]

    #------------

    digit_list_all = list(range(10))
    E_out_list_all_without_transform = []
    E_in_list_all_without_transform = []
    lambda_value = 1

    #------------

    min_E_out = np.inf
    min_digit = None

    for digit in digit_list_all:

        # train with training data!
        y_train = get_labels(digit, DIGIT_LABELS_TRAIN)
        w_tilde_reg = linear_regression_reg(Z_TRAIN, y_train, lambda_value)

        # compute E_in
        predicted_y_train = np.sign(np.dot(Z_TRAIN, w_tilde_reg)) 
        E_in = sum(predicted_y_train != y_train) / N_TRAIN
        E_in_list_all_without_transform.append(E_in)

        # compute E_out by using test data!
        predicted_y_test = np.sign(np.dot(Z_TEST, w_tilde_reg))
        y_test = get_labels(digit, DIGIT_LABELS_TEST)
        E_out = sum(predicted_y_test != y_test) / N_TEST
        E_out_list_all_without_transform.append(E_out)

        if E_out < min_E_out:
            min_E_out = E_out
            min_digit = digit


    print("\nQ9\nThe lowest out-of-sample error: \nE_out = {:0.5f}, \nis achieved for the {}-vs-all classifier".format(min_E_out, min_digit))

    plt.plot(digit_list_all, E_in_list_all_without_transform, 'ro-', label='E_in without transform')
    plt.plot(digit_list_all, E_out_list_all_without_transform, 'bo-', label='E_out without transform')
    plt.ylabel("out-of-sample error $E_{out}$")
    plt.xlabel("digit")
    plt.title("HW 9 - Problem 9 \n$E_{out}$ without transform ")
    plt.legend()
    plt.show()

    return E_in_list_all_without_transform, E_out_list_all_without_transform
    
    
E_in_list_all_without_transform, E_out_list_all_without_transform = problem9_E_out_without_transform()

# Compute the in-sample error E_out
def problem9_E_out_with_transform():

    x1 = X_TRAIN[:,0]
    x2 = X_TRAIN[:,1]
    Z_TRAIN = np.c_[np.ones(N_TRAIN), x1, x2, x1*x2, x1*x1, x2*x2]

    #------------

    x1 = X_TEST[:,0]
    x2 = X_TEST[:,1]
    Z_TEST = np.c_[np.ones(N_TEST), x1, x2, x1*x2, x1*x1, x2*x2]

    #------------

    digit_list_all = list(range(10))
    E_out_list_all_with_transform = []
    E_in_list_all_with_transform = []
    lambda_value = 1


    #------------

    min_E_out = np.inf
    min_digit = None

    for digit in digit_list_all:

        # train with training data!
        y_train = get_labels(digit, DIGIT_LABELS_TRAIN)
        w_tilde_reg = linear_regression_reg(Z_TRAIN, y_train, lambda_value)
        #print(w_tilde_reg.shape)

        # compute E_in
        predicted_y_train = np.sign(np.dot(Z_TRAIN, w_tilde_reg)) 
        E_in = sum(predicted_y_train != y_train) / N_TRAIN
        E_in_list_all_with_transform.append(E_in)

        # compute E_out by using test data!
        predicted_y_test = np.sign(np.dot(Z_TEST, w_tilde_reg))
        y_test = get_labels(digit, DIGIT_LABELS_TEST)
        E_out = sum(predicted_y_test != y_test) / N_TEST
        E_out_list_all_with_transform.append(E_out)

        if E_out < min_E_out:
            min_E_out = E_out
            min_digit = digit


    print("\nQ9\nThe lowest out-of-sample error: \nE_out = {:0.5f}, \nis achieved for the {}-vs-all classifier".format(min_E_out, min_digit))

    plt.plot(digit_list_all, E_in_list_all_with_transform, 'ro-', label='E_in with transform')
    plt.plot(digit_list_all, E_out_list_all_with_transform, 'bo-', label='E_out with transform')
    plt.ylabel("out-of-sample error $E_{out}$")
    plt.xlabel("digit")
    plt.title("HW 9 - Problem 9 \n$E_{out}$ with transform ")
    plt.legend()
    plt.show()
    
    return E_in_list_all_with_transform, E_out_list_all_with_transform
    
    
    
E_in_list_all_with_transform, E_out_list_all_with_transform = problem9_E_out_with_transform()

def problem9_comparison_E_out_with_and_without_transform():

    digit_list_all = list(range(10))
    plt.plot(digit_list_all, E_out_list_all_without_transform, 'go-', label='E_out without transform')
    plt.plot(digit_list_all, E_out_list_all_with_transform, 'mo-', label='E_out with transform')
    plt.ylabel("out-of-sample error $E_{out}$")
    plt.xlabel("digit")
    plt.title("HW 9 - Problem 9 \nComparing $E_{out}$ with and without transform ")
    plt.legend()
    plt.show()

    print("Let's plot the difference between the E_out values without and with transform:")
    difference_E_out_with_without_transform = np.array(E_out_list_all_without_transform) - np.array(E_out_list_all_with_transform)
    plt.plot(digit_list_all, difference_E_out_with_without_transform, '-ro')
    plt.xlabel("digit")
    plt.ylabel("$E_{out}$(without) - $E_{out}$(with)")
    plt.title("difference $E_{out}$ without and with transform")
    plt.show()


    print("difference: E_out without - E_out with transform:")
    for digit, E_out in zip(digit_list_all, difference_E_out_with_without_transform):
        print("digit = {} => difference = {}".format(digit, E_out))
        
problem9_comparison_E_out_with_and_without_transform()

"""
Exploring option 9[b], 9[c] and 9[d]
Conclusion: The out-of-sample error $E_{out}$ is smaller with a transform for digits 0, 1, 5. For the other digits the performance with and without transform is equal. We can exclude options 9[b], 9[c] and 9[d].

Exploring option 9[e]
For digit 5 there is an improvement of the out-of-sample error when using the transform. Let's check if the transform improves the out-of-sample performance by at least $5\%$ for 5-vs-all 
"""

print("ratio E_out_with / E_out_without = ", E_out_list_all_with_transform[5] / E_out_list_all_without_transform[5])
print("\nE_out_with <= 0.95 E_out_without ?", E_out_list_all_with_transform[5] <= 0.95 * E_out_list_all_without_transform[5])

"""
Exploring option 9[a]
Let's examine if overfitting occurs. First, let's plot how $E_{in}$ behaves with and without transform. Let's think about what we expect.

$E_{in}$ should go down if we use a more complex model, i.e. if we use the feature transform $\mathbf{z} = (1, x_1, x_2, x_1 x_2, x_1^2, x_2^2)$ which is more complex than the feature transform $\mathbf{z} = (1, x_1, x_2)$, then $E_{in}$ should go down.

The question is what happens with $E_{out}$ when we go from the less complex to the more complex model. For overfitting to occur $E_{out}$ should then go up.
"""

"""
Conclusion: We cannot observe overfitting, i.e. we cannot observe that when $E_{in}$ goes down, then $E_{out}$ goes up.
"""

# Q10
"""
Problem 10
We train the 1-vs-5 classifier with $\mathbf{z} = (1, x_1, x_2, x_1 x_2, x_1^2, x_2^2)$ for $\lambda = 0.01$ and $\lambda = 1$.
"""

df_train = pd.read_csv('D:\\AI\\lfd\\final\\features.train.txt', names = ['digit', 'intensity', 'symmetry'], sep='\s+', dtype=np.float64)

# Consider 1 vs 5 classifier
# We choose only rows with digits == 1 or digits == 5
df_train[df_train['digit'] == 1]
df_train[df_train['digit'] == 5]

# Append a column y with labels y=+1 for digit==1
ones = df_train[df_train['digit'] == 1].assign(y = np.ones(df_train[df_train['digit'] == 1].shape[0]))

# Append a column y with labels y=-1 for digit==5
# https://chrisalbon.com/python/pandas_assign_new_column_dataframe.html
fives = df_train[df_train['digit'] == 5].assign(y = -np.ones(df_train[df_train['digit'] == 5].shape[0]))

# Glue together the dataframes for ones and fives
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.append.html
df_1_vs_5 = ones.append(fives, ignore_index=True)

# Training data
X_train_1_vs_5 = np.c_[df_1_vs_5['intensity'], df_1_vs_5['symmetry']]

# labels
y_train_1_vs_5 = np.array(df_1_vs_5['y'])

N_train_1_vs_5 = X_train_1_vs_5.shape[0]

# Let's first read in the test set
df_test = pd.read_csv('D:\\AI\\lfd\\final\\features.test.txt', names = ['digit', 'intensity', 'symmetry'], sep='\s+', dtype=np.float64)

# Consider test set for 1-vs-5 classifier

# Append a column y with labels y=+1 for digit==1
ones_test = df_test[df_test['digit'] == 1].assign(y = np.ones(df_test[df_test['digit'] == 1].shape[0]))

# Append a column y with labels y=-1 for digit==5
# https://chrisalbon.com/python/pandas_assign_new_column_dataframe.html
fives_test = df_test[df_test['digit'] == 5].assign(y = -np.ones(df_test[df_test['digit'] == 5].shape[0]))


# Glue together the dataframes ones_test and fives_test
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.append.html
df_test_1_vs_5 = ones_test.append(fives_test, ignore_index=True)

# Create X_test
X_test_1_vs_5 = np.c_[df_test_1_vs_5['intensity'], df_test_1_vs_5['symmetry']]

y_test_1_vs_5 = np.array(df_test_1_vs_5['y'])

N_test_1_vs_5 = X_test_1_vs_5.shape[0]

# store the values for E_in for lambda = 1 and lambda = 0.01
def problem10():
    print("\nQ10\n")
    
    E_in_values = []
    E_out_values = []

    # feature matrix
    x1 = X_train_1_vs_5[:,0]
    x2 = X_train_1_vs_5[:,1]
    Z_TRAIN_1_vs_5 = np.c_[np.ones(N_train_1_vs_5), x1, x2, x1*x2, x1*x1, x2*x2]

    # feature matrix
    x1 = X_test_1_vs_5[:,0]
    x2 = X_test_1_vs_5[:,1]
    Z_TEST_1_vs_5 = np.c_[np.ones(N_test_1_vs_5), x1, x2, x1*x2, x1*x1, x2*x2]

    for lambda_value in [1, 0.01]:
        w_tilde_reg = linear_regression_reg(Z_TRAIN_1_vs_5, y_train_1_vs_5, lambda_value)

        # Compute E_in
        y_predict_train_1_vs_5 = np.sign(np.dot(Z_TRAIN_1_vs_5, w_tilde_reg))
        E_in = np.sum(y_train_1_vs_5 != y_predict_train_1_vs_5) / N_train_1_vs_5
        E_in_values.append(E_in)

        # Compute E_out
        y_predict_test_1_vs_5 = np.sign(np.dot(Z_TEST_1_vs_5, w_tilde_reg))
        E_out = np.sum(y_test_1_vs_5 != y_predict_test_1_vs_5) / N_test_1_vs_5
        E_out_values.append(E_out)
        
    return E_in_values, E_out_values
    
    
E_in_values, E_out_values = problem10()

print("E_in values: ", E_in_values)
print("E_out values: ", E_out_values)

print("\nE_in_lambda_1 == E_in_lambda_0_01 ?", E_in_values[0] == E_in_values[1])
print("E_out_lambda_1 == E_out_lambda_0_01 ?", E_out_values[0] == E_out_values[1])

# Conclusion We can exclude options 10[b] and 10[c].

# 10[a]: Check if overfitting occurs from $\lambda = 1$ to $\lambda = 0.01$.

plt.plot(E_in_values, 'ro-', label='E_in')
plt.plot(E_out_values, 'bo-', label='E_out')
plt.title("Does overfitting occur? \nThe data pair on the left is for $\lambda=1$,\nthe data pair on the right is for $\lambda=0.01$")
plt.legend()
plt.show()

"""
Conclusion: We can see that $E_{out}$ goes up, while $E_{in}$ goes down. This means that overfitting occurs from $\lambda = 1$ to $\lambda = 0.01$

This means the correct answer is 10[a].

The plot above also let's us exclude options 10[d] and 10[e].

Result for Problem 10
The correct answer is 10[a].
"""
# ANSWER = 10[a]


















