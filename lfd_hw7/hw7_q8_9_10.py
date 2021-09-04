# -*- coding: utf-8 -*-
"""
Created on: Tue Jul  6 15:04:18 2021

@author: Mahmut Osmanovic
"""
# In the following problems, we compare PLA to SVM with hard margin1 on linearly
# separable data sets.

# Amount of times that g_svm is better than g_pla in aprx f

# import libraries
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import time

# implement PLA
def PLA(Z, y_f):
    '''
    - Takes 
      Z: feature matrix, 
      y_f: labels by target function f
    - Returns weight vector w using the perceptron learning algorithm 
    '''
    
    num_features =  Z.shape[1]
    w_h = np.zeros(num_features)                    # initialize weight vector for hypothesis h
    t = 0                                           # count number of iterations in PLA
    
    while True:
        # Start PLA
        y_h = np.sign(np.dot(Z, w_h))               # classification by hypothesis
        comp = (y_h != y_f)                         # compare classification with actual data from target function
        wrong = np.where(comp)[0]                   # indices of points with wrong classification by hypothesis h

        if wrong.size == 0:
            break
        
        rnd_choice = np.random.choice(wrong)        # pick a random misclassified point

        # update weight vector (new hypothesis):
        w_h = w_h +  y_f[rnd_choice] * np.transpose(Z[rnd_choice])
        t += 1
        if t == 10**3:
            break
    
    return w_h

def get_TF():
    
    """
    - Returns weight vector w_f defining the target function
        which is a line through points A and B
    """
    
    # returns a vector of n points from the interval [-1,1]
    rnd = lambda n: np.random.uniform(-1,1,n)
    
    # choose two random points A, B in [-1,1] x [-1,1] 
    A = rnd(2)
    B = rnd(2)
    
    # the line can be describe by y = kx + m,
    #   k = slope, m = intercept with y axis
    k = (B[1] - A[1]) / (B[0] - A[0])
    m = B[1] - k * B[0]  
    w_f = np.array([m, k, -1])  
    
    return w_f

def get_target_labels(Z, w_f):
    """
    - Takes feature matrix Z
    - Takes weight vector w_f of target function
    - Returns labels by target function f
    """
    return np.sign(np.dot(Z, w_f))

def get_random_points(N):
    """
    - Takes number of parameters N
    - Returns tuple (x1,x2), where x1 and x2 are vectors
    """    
    x1 = np.random.uniform(-1,1,N)
    x2 = np.random.uniform(-1,1,N)
    return (x1,x2)

def weight_to_line(w):
    """
    - Takes weight vector w
    - Returns slope m and intercept b of line
    
    We know that w^T x = 0 for points (x1,x2) on the line, so
    w0 * 1 + w1 * x1 + w2 * x2 = 0
    w0     + w1 * x1 + w2 * x2 = 0
    
    Let's solve for x2:
        w2 * x2 = -w0 - w1*x1
    =>       x2 = -(w0/w2) + (-w1/w2) * x1
    
    This is the equation of a line x2 = b + m*x1.
    The intercept is b = -(w0/w2),
    the slope is m = -w1/w2
    
    Let's return both
    """
    b = -(w[0] / w[2])
    m = -(w[1] / w[2])
    
    return b, m


def predict(Z_test, w):
    '''
    - Takes feature matrix Z_test
    
    - Takes weight vector obtained from final hypothesis g
      (learned model).
      
    - Returns classification vector y via sign of 
      linear classifier sign(w^T z)
    '''
    y = np.sign(np.dot(Z_test, w))
    return y


def get_error(y_test, y_predict):
    '''
    - Takes labels of test set y_test (true labels)
    - Takes predicted labels (classifications) by model
    - Returns error, i.e. fraction of points whose predicted label does not equal
      the true label
    '''
    N_test = y_test.shape[0]
    return np.sum(y_test != y_predict) / N_test
    
def problem_8(N_train, RUNS=1000, SHOW_PLOT_TRAIN=False, SHOW_PLOT_TEST=False, PRINT_ERRORS=False):
    #RUNS = 1000
    iteration = RUNS
    
    #N_train = 10
    N_test = 1000
    
    #SHOW_PLOT_TRAIN = False
    #SHOW_PLOT_TEST = False
    #PRINT_ERRORS = False
    
    E_test_PLA_total = 0
    E_test_SVM_total = 0
    SVM_better_than_PLA_total = 0
    num_SVM_vectors_total = 0
    
    while iteration:
        
        # -----------------------------------
        
        # TRAINING
        x1_train, x2_train = get_random_points(N_train)
        
        # feature matrix for training set
        Z_train = np.c_[np.ones(N_train), x1_train, x2_train]
        
        # weight vector of traget function f
        w_f = get_TF()
        
        # labels by target function f
        y_train = get_target_labels(Z_train, w_f)
        
        # discard if all classified the same
        # "If all data points are on one side of the line, discard the run and start a new run"
        if abs(sum(y_train)) == N_train:
            continue
        
        # PLA -------------------------------
        
        w_PLA = PLA(Z_train, y_train)
        
        # -----------------------------------
        
        # SVM classifier clf
        clf = svm.SVC(C = np.inf, kernel = 'linear')
        clf.fit(Z_train[:, 1:], y_train)
        
        w_SVM = np.array(3*[None])
        w_SVM[1], w_SVM[2] = clf.coef_[0]
        w_SVM[0] = clf.intercept_
        num_SVM_vectors = sum(clf.n_support_)
        
        num_SVM_vectors_total += num_SVM_vectors
        
        # Computing the error on in-sample training set
        # this error should be zero! This is a check.
        # Predicitions
        predict_PLA_train = predict(Z_train, w_PLA)
        predict_SVM_train = predict(Z_train, w_SVM)
        E_in_PLA = get_error(y_train, predict_PLA_train)
        E_in_SVM = get_error(y_train, predict_SVM_train)
        
        # -----------------------------------
        # -----------------------------------
        
        # TESTING
        x1_test, x2_test = get_random_points(N_test)
        
        # feature matrix for training set
        Z_test = np.c_[np.ones(N_test), x1_test, x2_test]
        
        # labels by target function f
        y_test = get_target_labels(Z_test, w_f)
        
        # predictions
        predict_PLA = predict(Z_test, w_PLA)
        predict_SVM = predict(Z_test, w_SVM)
        
        # errors
        E_test_PLA = get_error(y_test, predict_PLA)
        E_test_SVM = get_error(y_test, predict_SVM)
        
        E_test_PLA_total += E_test_PLA
        E_test_SVM_total += E_test_SVM
        
        #print("E_test_PLA = ", E_test_PLA)
        #print("E_test_SVM = ", E_test_SVM)
        
        # Note: Smaller errorrate is better
        SVM_better_than_PLA_total += (E_test_SVM < E_test_PLA)
        
        if PRINT_ERRORS:
            print("E_in_PLA = ", E_in_PLA)
            print("E_in_SVM = ", E_in_SVM)
            print("E_test_PLA = ", E_test_PLA)
            print("E_test_SVM = ", E_test_SVM)            
    
        #----------------------------------
        
        if SHOW_PLOT_TRAIN:
            print("\nShowing plot for training set:")
            
            # plot points classified by target function f
            plt.scatter(x1_train[y_train==1], x2_train[y_train==1], color='lightsteelblue')
            plt.scatter(x1_train[y_train==-1], x2_train[y_train==-1], color='lightpink')

            # points on the line of the target function
            intercept, slope = weight_to_line(w_f)
            A = [-1, intercept + slope*(-1)]
            B = [1, intercept + slope*(1)]
            plt.plot([A[0], B[0]], [A[1], B[1]], 'g', label='Target')

            # points on the line of the PLA hypothesis
            intercept_PLA, slope_PLA = weight_to_line(w_PLA)
            A_PLA = [-1, intercept_PLA + slope_PLA*(-1)]
            B_PLA = [1, intercept_PLA + slope_PLA*(1)]
            plt.plot([A_PLA[0], B_PLA[0]], [A_PLA[1], B_PLA[1]], 'b:', label='PLA')

            # points on the line of the SVM hypothesis
            intercept_SVM, slope_SVM = weight_to_line(w_SVM)
            A_SVM = [-1, intercept_SVM + slope_SVM*(-1)]
            B_SVM = [1, intercept_SVM + slope_SVM*(1)]
            plt.plot([A_SVM[0], B_SVM[0]], [A_SVM[1], B_SVM[1]], 'r-.', label='SVM')
            
            #----------------------
            
            plt.xlim(-1,1)
            plt.ylim(-1,1)
        
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.title('Problem 8')
            plt.legend()
            plt.show()
            print('--')
            
        #----------------------------------    
    
        if SHOW_PLOT_TEST:
            print("\nShowing plot for testing set:")
            
            # plot test points classified by target function f
            plt.scatter(x1_test[y_test==1], x2_test[y_test==1], color='lightsteelblue')
            plt.scatter(x1_test[y_test==-1], x2_test[y_test==-1], color='lightpink')

            # points on the line of the target function
            intercept, slope = weight_to_line(w_f)
            A = [-1, intercept + slope*(-1)]
            B = [1, intercept + slope*(1)]
            plt.plot([A[0], B[0]], [A[1], B[1]], 'g', label='Target')

            # points on the line of the PLA hypothesis
            intercept_PLA, slope_PLA = weight_to_line(w_PLA)
            A_PLA = [-1, intercept_PLA + slope_PLA*(-1)]
            B_PLA = [1, intercept_PLA + slope_PLA*(1)]
            plt.plot([A_PLA[0], B_PLA[0]], [A_PLA[1], B_PLA[1]], 'b:', label='PLA')

            # points on the line of the SVM hypothesis
            intercept_SVM, slope_SVM = weight_to_line(w_SVM)
            A_SVM = [-1, intercept_SVM + slope_SVM*(-1)]
            B_SVM = [1, intercept_SVM + slope_SVM*(1)]
            plt.plot([A_SVM[0], B_SVM[0]], [A_SVM[1], B_SVM[1]], 'r-.', label='SVM')
            
            #----------------------
            
            plt.xlim(-1,1)
            plt.ylim(-1,1)
        
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.title('Problem 8')
            plt.legend()
            plt.show()
            
            print('----------------------------------------------------')
            
        
        iteration -= 1
        
        
    E_test_PLA_avg = E_test_PLA_total / RUNS
    E_test_SVM_avg = E_test_SVM_total / RUNS
    SVM_better_than_PLA_avg = SVM_better_than_PLA_total / RUNS
    num_SVM_vectors_avg = num_SVM_vectors_total / RUNS
    
    return E_test_PLA_avg, E_test_SVM_avg, SVM_better_than_PLA_avg, num_SVM_vectors_avg    

""" QUESTION 8: [c] ~= 0.593-0.598, closest to 60 %
# set N_train = 10
start_time = time.time()
E_test_PLA_avg, E_test_SVM_avg, count_SVM_better_than_PLA, num_SVM_vectors_avg = problem_8(N_train=10, RUNS=1000)
print("--- %s seconds ---" % (time.time() - start_time))
print("count_SVM_better_than_PLA = ", count_SVM_better_than_PLA)    
"""

"""
#QUESTION 9: [d], about 0.613-0.614, ans closest to 70 %
# set N_train = 10
start_time = time.time()
E_test_PLA_avg, E_test_SVM_avg, count_SVM_better_than_PLA, num_SVM_vectors_avg = problem_8(N_train=100, RUNS=1000)
print("--- %s seconds ---" % (time.time() - start_time))
print("count_SVM_better_than_PLA = ", count_SVM_better_than_PLA)    
"""   

"""
#QUESTION 10: [b], about 2.848 ~= 3
# set N_train = 10
start_time = time.time()
E_test_PLA_avg, E_test_SVM_avg, count_SVM_better_than_PLA, num_SVM_vectors_avg = problem_8(N_train=10, RUNS=1000)
print("--- %s seconds ---" % (time.time() - start_time))
print("num_SVM_vectors_avg = ", num_SVM_vectors_avg)    
"""
    
    
    
    
    