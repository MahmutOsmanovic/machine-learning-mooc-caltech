# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 14:23:37 2021

@author: Mahmu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# https://stackoverflow.com/questions/9777783/suppress-scientific-notation-in-numpy-when-creating-array-from-nested-list
np.set_printoptions(suppress=True)

data_in = np.loadtxt("D:\\AI\\lfd\\lfd_hw7\\in.dta.txt")


training = data_in[:25, :]

x1_train = training[:,0]
x2_train = training[:,1]
y_train = training[:,2]

validation = data_in[25:, :]
print(validation)

x1_val = validation[:,0]
x2_val = validation[:,1]
y_val = validation[:,2]
N_val = validation.shape[0]

def get_feature_matrix_Z_k(x1, x2, k):
    '''
    - Takes vectors x1 and x2
    - builds new feature matrix Z in the transformed space,
    with up to 8 features z0, z1, ..., z7
    - parameter k determines model
    - returns feature matrix using columns z0, z1, ..., zk
    '''
    N = x1.shape[0]
    Z = np.array([np.ones(N), x1, x2,
                  x1**2, x2**2, x1*x2,
                  np.absolute(x1-x2), np.absolute(x1+x2)]).T
    
    # Using only columns 0, 1, ... , k
    return Z[:, :(k+1)]
    
#-----------------------------------------------------------
    
def linear_regression(x1, x2, y, k):
    '''
    - Takes vector x1, vector x2, vector y
    - parameter k
    - returns weight vector for linear regression
      using vectors of the form (z0, z1, ..., zk)
    '''
    
    # feature matrix Z_k
    N = x1.shape[0]
    Z_k = get_feature_matrix_Z_k(x1, x2, k)
    
    # see lecture 3, slide 17
    Z_dagger = np.dot(np.linalg.inv(np.dot(Z_k.T, Z_k)), Z_k.T)

    # Use linear regression to get weight vector
    w_tilde = np.dot(Z_dagger, y)
    
    return w_tilde

# perform linear regression for k = 3, ..., 7

w_tildes = 8 * [None]
for k in range(1, 8):
    w_tildes[k] = linear_regression(x1_train, x2_train, y_train, k)

def predict(x1_test, x2_test, w_tilde_k):
    '''
    - Takes vectors x1_test, x2_test corresponding 
    to unseen points (x1_test, x2_test)
    - Takes hypothesis / model w_tilde_k
    - Returns predictions for these points using
    the hypothesis w_tilde_k
    '''
    
    k = w_tilde_k.shape[0] - 1     # length of w is k + 1, so k = w-1
    Z_k_test = get_feature_matrix_Z_k(x1_test, x2_test, k)
    return np.sign(np.dot(Z_k_test, w_tilde_k))

errors_val = 8 * [None]
predictions_k_val = 8 * [None]

for k in range(1, 8):
    errors_val[k] = sum(y_val != predict(x1_val, x2_val, w_tildes[k])) / N_val
    predictions_k_val[k] = predict(x1_val, x2_val, w_tildes[k])
    
for k in range(1, 8):
    print("k=", k, "    => E_val =", errors_val[k])
    
# plot points in validation set
plt.plot(x1_val[y_val==1], x2_val[y_val==1], 'ro', label='$y=+1$')
plt.plot(x1_val[y_val==-1], x2_val[y_val==-1], 'bo', label='$y=-1$')

title_string = "Validation set, $N_{val} = 10$"
plt.title(title_string)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()

u = np.arange(-1.0,1.0,0.02)
X,Y= np.meshgrid(u,u)


# view validation test set
for k in range(1, 8):
    
    fig = plt.figure(k, dpi = 80)
    
    # plot points
    plt.plot(x1_val[y_val==1], x2_val[y_val==1], 'ro', label='$y=+1$')
    plt.plot(x1_val[y_val==-1], x2_val[y_val==-1], 'bo', label='$y=-1$')
    
    # plot correctly classified as blue and misclassified as red
    misclassified = (y_val != predictions_k_val[k])
    plt.plot(x1_val[misclassified], x2_val[misclassified], 'mo', label='misclassified')
    
    #print(w_tildes[k])
    w = list(w_tildes[k]) + (8-k-1) * [0]

    # plot decision boundary
    boundary = lambda x1, x2, w: w[0]*1 + w[1]*x1 + w[2]*x2 + w[3]*x1**2 + w[4]*x2**2 + w[5]*x1*x2 +w[6]* np.absolute(x1-x2) +w[7]* np.absolute(x1+x2)
    phi = boundary(X,Y, w)
    plt.contour(X,Y,phi, [0.0], colors = 'g')

    
    title_string = "Classification of validation set\n$N_{train}=25, N_{val}=10," + " k={0}, $".format(str(k))
    title_string += ("$E_{val}=$" + str(errors_val[k]))
    plt.title(title_string)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.show()
    

data_out = np.loadtxt("D:\\AI\\lfd\\lfd_hw7\\out.dta.txt")
df_out = pd.DataFrame(data_out, columns=['x1', 'x2','y'])

x1_test = data_out[:,0]
x2_test = data_out[:,1]
y_test = data_out[:,2]

N_test = data_out.shape[0]
    
errors_test = 8 * [None]
predictions_k_test = 8 * [None]

for k in range(1, 8):
    errors_test[k] = sum(y_test != predict(x1_test, x2_test, w_tildes[k])) / N_test
    predictions_k_test[k] = predict(x1_test, x2_test, w_tildes[k])
    
print("\n")    
for k in range(1, 8):
    print("k=", k, "    => E_out =", errors_test[k])
    
# plot points in test set
plt.plot(x1_test[y_test==1], x2_test[y_test==1], 'ro', label='$y=+1$')
plt.plot(x1_test[y_test==-1], x2_test[y_test==-1], 'bo', label='$y=-1$')

title_string = "Test set, $N_{val} = 250$"
plt.title(title_string)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()

# use 'out.dta.txt' as test set
for k in range(1, 8):
    

    fig = plt.figure(k, dpi = 80)
    #plt.plot(x1_val[y_val==1], x2_val[y_val==1], 'ro')
    #plt.plot(x1_val[y_val==-1], x2_val[y_val==-1], 'bo')
    
    # plot points
    plt.plot(x1_test[y_test==1], x2_test[y_test==1], 'ro', label='$y=+1$')
    plt.plot(x1_test[y_test==-1], x2_test[y_test==-1], 'bo', label='$y=-1$')
    
    # plot correctly classified as blue and misclassified as red
    misclassified = (y_test != predictions_k_test[k])
    plt.plot(x1_test[misclassified], x2_test[misclassified], 'mo', label='misclassified')
    
    #print(w_tildes[k])
    w = list(w_tildes[k]) + (8-k-1) * [0]
    #print(w)

    boundary = lambda x1, x2, w: w[0]*1 + w[1]*x1 + w[2]*x2 + w[3]*x1**2 + w[4]*x2**2 + w[5]*x1*x2 +w[6]* np.absolute(x1-x2) +w[7]* np.absolute(x1+x2)
    phi = boundary(X,Y, w)
    plt.contour(X,Y,phi, [0.0], colors = 'g')

    
    title_string = "Classification of test set\n$N_{train}=25, N_{val}=10, N_{test}=250," + " k={0}, $".format(str(k))
    title_string += ("$E_{test}=$" + str(errors_test[k]))
    plt.title(title_string)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.show()