# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 05:11:52 2021

@author: Mahmu
"""

import numpy as np
import pylab
import random
# 1.

def runQ1(sigma, d, E_in): 
    #E_in = s**2(1-d+1/N)
    #E_in/s**2=1 - d+1/N
    #d+1/N = 1 - E_in/s**2
    N = (d+1) / (1 - E_in/(sigma**2))
    text = "Atleast: " + str(N) + " examples"
    print(text)

# runQ1(0.1,8,0.008)
# print("Answer: [c]")

# 2.

def runQ2(w0,w1,w2):
    x1 = np.linspace(-10,10, 1000)
    
    # a)
    # 0 = w0 + w1*(x1**2) + w2*(x2**2)
    x2 = np.sqrt(-(w0 + w1*(x1**2))/w2)    
    
    pylab.plot(x1, x2)

# runQ2(-1,0,3) No. a)
# runQ2(1,2,0) No. b)
# runQ2(-1,2,2) No. c)
# runQ2(20,-2,2) Yes.
# runQ2(-20,2,-2) Maybe.

# -> Answer depends on sign of w0.
# For hyperbola like desired one: https://sv.wikipedia.org/wiki/Hyperbel w0>0

# 3.

# Fi(x) = 15 outputs
# d_vc = d + 1 = {x1,x2} = 2 + 1 = 3 (2D perceptron)
# After transformation: d_vc = dz + 1 = 15 (number of output parameters of fi)
# -> 15 is NOT smaller than 15 -> ANSWER: 15 
    
# 4. Partial derivative of E(u, v) = (ue**(v) − 2ve**(−u))**2 w.r.t. u

# ANSWER =  dE/du = 2(ue**(v) − 2ve**(−u))(e**(v) + 2ve**(-u)) -> [e]

# 5. How many iterations does it take for the error E(u,v) to fall below 10**(-14) 
#    for the first time? (Use double precision)

# Starting point: (u,v) = 1,1
# Minimize error
# Learning rate: 0.1: n = LR * grad(E)

def E(x):
    u = x[0]
    v = x[1]
    return (u*np.exp(v) - 2*v*np.exp(-u))**2

def gradE(x):
    u = x[0]
    v = x[1]
    return np.array([
        2*(u*np.exp(v) - 2*v*np.exp(-u))*(np.exp(v) + 2*v*np.exp(-u)),
        2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v) - 2*np.exp(-u))
        ])

    # eta is the learning rate
    # tol is the tolerance        
def gradient_descent(x0, tol=1e-14, eta=0.1):
    x = x0.copy()
    its = 0
    err = E(x)
    us = [x[0]]
    vs = [x[1]]
    while err > tol:
        grad = gradE(x)
        x = x - eta*grad
        us.append(x[0])
        vs.append(x[1])
        err = E(x)
        its += 1
        if its > 100:
            print("Didn't converge")
            return
    return (x, err, its, us, vs)
    
def runQ4(x0, tol=1e-14, eta=0.1):
    final_uv, err, its, us, vs = gradient_descent(x0)
    print("Final coordinate", final_uv)
    print("Final err:", err)
    print("Iterations:", its)
    pylab.plot(us,vs, ".-")
    pylab.show()
    pylab.plot(us[3:], vs[3:], ".-")
    pylab.show()

x = np.array([1, 1])
#runQ4(x)    

# -> Answer: 5 is D, 10 iterations. Closest coordinate is E, (0.045,0.024)

# 7.

def dEdu(u,v):
    return 2*(u*np.exp(v) - 2*v*np.exp(-u))*(np.exp(v) + 2*v*np.exp(-u))

def dEdv(u, v):
    return 2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v) - 2*np.exp(-u))

def coordinate_desecent(u0, v0, tol=1e-14, eta=0.1):
    u, v = u0, v0
    us = [u]
    vs = [v]
    for it in range(15):
        u=u-eta*dEdu(u,v)
        us.append(u)
        vs.append(v)
        v=v-eta*dEdv(u,v)
        us.append(u)
        vs.append(v)
    return u,v,E([u,v]),15,us,vs

def runQ7():
    final_u, final_v, err, its, us, vs = coordinate_desecent(1, 1)
    print("Final coordinate:", final_u, final_v)
    print("Final err:", err)
    print("Iterations", its)
    pylab.plot(us,vs,".-")
    pylab.show()
    pylab.plot(us[10:], vs[10:], ".-")
    pylab.show()
    
#runQ7()

def E3(u,v):
    return (u*np.exp(v) - 2*v*np.exp(-u))**2 

# 8.

""" Logistic Regression
In this problem you will create your own target function f (probability in this case)
and data set D to see how Logistic Regression works. For simplicity, we will take f
to be a 0/1 probability so y is a deterministic function of x.
Take d = 2 so you can visualize the problem, and let X = [−1, 1]×[−1, 1] with uniform
probability of picking each x ∈ X . Choose a line in the plane as the boundary between
f(x) = 1 (where y has to be +1) and f(x) = 0 (where y has to be −1) by taking two
random, uniformly distributed points from X and taking the line passing through
4
them as the boundary between y = ±1. Pick N = 100 training points at random
from X , and evaluate the outputs yn for each of these points xn.
Run Logistic Regression with Stochastic Gradient Descent to find g, and estimate Eout
(the cross entropy error) by generating a suffic iently large, separate set of points to
evaluate the error. Repeat the experiment for 100 runs with different targets and take
the average. Initialize the weight vector of Logistic Regression to all zeros in each
run. Stop the algorithm when kw(t−1) − w(t)k < 0.01, where w(t) denotes the weight
vector at the end of epoch t. An epoch is a full pass through the N data points (use a
random permutation of 1, 2, · · · , N to present the data points to the algorithm within
each epoch, and use different permutations for different epochs). Use a learning rate
of 0.01.
8. Which of the following is closest to Eout for N = 100?
[a] 0.025
[b] 0.050
[c] 0.075
[d] 0.100
[e] 0.125
"""

d = 2
N = 100

def random_point():
    return 2*np.random.rand()-1, 2*np.random.rand()-1

def getDataPoints(points, k, m):
    R = 2*np.random.rand(points,3) # (points)x(3) array, pts btw 0 n 2
    X = R - 1 # pts, btw -1 n 1 (acc to question req)
    X[:,0] = 1 # first point
    y = np.sign(m + k*X[:,1] - X[:,2])
    return X,y
    
def TF():
    x1,y1 = random_point()
    x2,y2 = random_point()
    k = (y2-y1)/(x2-x1)
    m = y2 - k*x1
    return k, m

def drawAllTheStuff():
    k, m = TF()
    X, y = getDataPoints(N, k, m)
    ibelow = np.where(y==-1)
    iabove =  np.where(y==1)
    
    pylab.scatter(X[ibelow,1], X[ibelow,2], c="red")
    pylab.scatter(X[iabove,1], X[iabove,2], c="blue")
    pylab.show()
    

#drawAllTheStuff()

def sigmoid(X):
    return 1 / (1 + np.exp(-x))

class LogisticModel:
    
    def fit(self, X, y):
        eta = 0.01
        N = X.shape[0]
        w = np.zeros(X.shape[1])
        prev_w = np.ones(X.shape[1])
        epoch = 0
        
        while True:
            prev_w = w
            for n in np.random.permutation(N):
                n = np.random.randint(0,N)
                gradE = -y[n]*X[n,:] / (1 + np.exp(y[n] * w.dot(X[n,:])))
                w = w - eta * gradE
                
            epoch += 1
            if np.linalg.norm(w - prev_w) < 0.01:
                self.w = w
                break
            
        return epoch
    
    
    def predict(self, X):
        return np.sign(X @ self.w)
    
    def cross_entropy_error(self, X, y):
        N = X.shape[0]
        errs = []
        for n in range(N):
            errs.append(np.log(1 + np.exp(-y[n] * self.w.dot(X[n,:]))))
        return np.mean(errs)

    
num_epochs = []
cross_entropy_errors = []
    
def runQ8():
    for experiment in range(1):
        k, m = TF()
        X,y = getDataPoints(N, k, m)
        lm = LogisticModel()
        epochs = lm.fit(X,y)
        
        X_test, y_test = getDataPoints(20*N, k, m)
        y_predicted = lm.predict(X_test)
        
        err = lm.cross_entropy_error(X_test, y_test)
        num_epochs.append(epochs)
        cross_entropy_errors.append(err)
        
        print("Epochs: ", np.mean(num_epochs))
        print("Error: ", np.mean(cross_entropy_errors))
        

    ibelow = np.where(y_predicted == -1)
    iabove = np.where(y_predicted == 1)
    pylab.scatter(X_test[ibelow,1], X_test[ibelow, 2])
    pylab.scatter(X_test[iabove,1], X_test[iabove, 2])
    pylab.title("Predicted")
    pylab.show()
    
    ibelow = np.where(y_test == -1)
    iabove = np.where(y_test == 1)
    pylab.scatter(X_test[ibelow, 1], X_test[ibelow, 2])
    pylab.scatter(X_test[iabove,1], X_test[iabove,2])
    pylab.title("Actual")
    pylab.show()
        
runQ8()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    