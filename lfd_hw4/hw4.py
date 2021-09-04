# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 18:05:09 2021

@author: Mahmu
"""

# hw4 
import pylab
import numpy as np
import random as rd
import scipy.integrate as integrate
import scipy

# 1.
d_vc = 10
wanted_probability = 0.95
eps = 0.05
# N = ?

def mH(N):
    return N**(d_vc)

def runQ1():
     N = np.arange(10*d_vc, 500000)
     currProb = 4*mH(2*N)*np.exp(- 1/8 * eps*eps*N)
     pylab.plot(N, currProb, c = 'b')
     pylab.plot([0, 500000], [0.05, 0.05], c='r')
     
     pylab.ylim(-1,1)
     pylab.xlim(0,500000)
 
#runQ1()

def runQ1Try2():
    N = np.arange(10*d_vc, 500000)
    conf = 4*np.exp(d_vc*np.log(2*N) - 1/8 * eps*eps*N)

    pylab.plot(N, conf)
    pylab.plot([0, 500000], [0.05, 0.05])
    pylab.ylim([0, 1])
    pylab.show()

    N = np.arange(440000, 460000)
    conf = 4*np.exp(d_vc*np.log(2*N) - 1/8 * eps*eps*N)

    pylab.plot(N, conf)
    pylab.plot([440000, 460000], [0.05, 0.05])
    pylab.ylim([0, 1])

#runQ1Try2()

"""
Now we're getting somewhere. Error in last exercise was probably due to NumPy using some kind of integer type 
that wrapped around to negative numbers when getting too large. Simplifying the expression did the trick and 
prevented numerical issues.
"""

# 2. Smallest bound on eps, the generalization error? (approx error)

d_vc = 50
delta = 0.05

def vc_bound(N):
    return np.sqrt(8/N * np.log(4*mH(2*N)/delta))

def rademacher_pentaly_bound(N):
    return np.sqrt(2*np.log(2*N*mH(N))/N) + np.sqrt(2/N*np.log(1/delta)) + 1/N

def parrondo_and_Van_den_Broek(N):
    # Solve implicit expression for epsilon using quadratic formula
    return 1/N + 0.5*np.sqrt(4/N**2 + 4/N * np.log(6*mH(2*N)/delta))

def devroye(N):
    # Solve implicit expression for epsilon using quadratic formula
    # Need to fix so that numbers fit in float64
    #return (2/N + np.sqrt(4/N**2 + 4*(1-2/N)*(1/(2*N))*np.log(4*mH(N**2)/delta)))/(2*(1-2/N))
    return (2/N + np.sqrt(4/N**2 + 4*(1-2/N)*(1/(2*N))*(np.log(4) + 2*d_vc*np.log(N) - np.log(delta))))/(2*(1-2/N))

def runQ2():
    N = np.arange(9900, 10000, dtype = np.float64)
    
    pylab.plot(N,vc_bound(N), c = 'blue', label = 'Original VC Bound')
    pylab.plot(N,rademacher_pentaly_bound(N), c = 'orange', label = 'Rademacher Pentaly Bound')
    pylab.plot(N,parrondo_and_Van_den_Broek(N), color = 'green', label = 'Parrondo and Van den Broek')
    pylab.plot(N,devroye(N), color = 'red', label = 'Devroye')
    
    
    pylab.ylim([0.19,0.66])
    pylab.legend()
    pylab.xlabel("Amount of example, N")
    pylab.ylabel("Generalization error, ε")
    pylab.title("Generalization error as a function of number of examples")
    pylab.show()
    
runQ2()

# 3. Same d_vc, same delta, small N, say, N = 5

def runQ3():
    N = np.arange(3, 10, dtype = np.float64)
    
    pylab.plot(N,vc_bound(N), ".-", c = 'blue', label = 'Original VC Bound')
    pylab.plot(N,rademacher_pentaly_bound(N), ".-", c = 'orange', label = 'Rademacher Pentaly Bound')
    pylab.plot(N,parrondo_and_Van_den_Broek(N), ".-", color = 'green', label = 'Parrondo and Van den Broek')
    pylab.plot(N,devroye(N), ".-", color = 'red', label = 'Devroye')
    
    
    pylab.ylim([3,18])
    pylab.legend()
    pylab.xlabel("Amount of example, N")
    pylab.ylabel("Generalization error, ε")
    pylab.title("Generalization error as a function of number of examples")
    pylab.show()
    
#runQ3()

""" f(x) = sin(pi * x), where TF: f:[-1,-1] -> R, input probability distribution is uniform on [-1,1]
    The training set consists of two independently picked points. 
    The learning algorithm produces the hypothesis that minimizes the mean squared error on the examples."""

# 4. h(x) = kx, g(x) = ?

def getK():
    x1 = rd.uniform(-1,1)
    x2 = rd.uniform(-1,1)
    y1 = np.sin(x1*np.pi)
    y2 = np.sin(x2*np.pi)
    return (y2-y1)/(x2-x1) 

def getX_Sample_and_f_x(test_points):
     x_sample = []
     f_x = []
     for i in range(test_points):
         x = rd.uniform(-1, 1)
         y = np.sin(x*np.pi)
         x_sample.append(x)
         f_x.append(y)
     return x_sample, f_x
 
def getSquaredErrorList(x_sample, f_x, test_points, k):
    listOfErr = []
    for i in range(test_points):
        sqrd_err = (k*x_sample[i]-f_x[i])**(2)
        listOfErr.append(sqrd_err)
    return listOfErr

def getErrorForH(test_points,hypothesisCount):
    hypothesisSet = []
    Err = []
    for h in range(hypothesisCount):
        k = getK()
        x_sample, f_x = getX_Sample_and_f_x(test_points)
        listOfErr = getSquaredErrorList(x_sample, f_x, test_points, k)        
        Err.append(np.mean(listOfErr))
        hypothesisSet.append(k)
    return hypothesisSet, Err
    
def getG(hypothesisSet, Err):
    indexErr = Err.index(min(Err))
    g = hypothesisSet[indexErr]
    return g    

def drawAll(g):
    x = np.linspace(-1,1,100)
    pylab.plot([-1, 1],[0, 0], '--', c = 'k')
    pylab.plot([0, 0],[-1, 1], '--', c = 'k')
    pylab.plot(x, np.sin(np.pi*x), c = 'b', label = 'Target function')
    pylab.plot(x, g*x, c = 'r', label = 'g = ' + str(round(g,2)) + 'x')
    pylab.legend()
    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.title("Best linear approximation to f(x) using LSM")
    pylab.show()    
    

def runQ4():
    test_points = 1000
    hypothesisCount = 1000
        
    hypothesisSet, Err = getErrorForH(test_points,hypothesisCount)
    g = getG(hypothesisSet, Err)  # g is the best hypothesis "h" in the hypothesis set
    
    drawAll(g)
    
    print("error", np.mean(Err))
        
runQ4() 

# 5 and 6. What is the closest value to the bias and variance? (Redo 4)

def f(x):
    return np.sin(np.pi*x)

def error(k):
    return integrate.quad(lambda x: (f(x)-k*x)**2, -1, 1)[0]

def runQ4_2():
    k_range = np.linspace(-5,5)
    errs = [error(k) for k in k_range]
    pylab.plot(k_range, errs)
    pylab.xlabel("k")
    pylab.ylabel("error")
    pylab.show()
    
    optim = scipy.optimize.minimize_scalar(error,bounds=(-5,5))
    print(optim)
    
    x = np.linspace(-1, 1)
    pylab.fill_between(x, f(x), optim.x*x)
    pylab.legend(["Error"])
    pylab.title("k with minimum error = g, for dataset D")
    pylab.show()
    
    gg = []
    for i in range(10000):
        x1 = rd.uniform(-1,1)
        x2 = rd.uniform(-1,1)
        X = np.array([[x1], [x2]])
    
        y1 = f(x1)
        y2 = f(x2)
        y = np.array([[y1],[y2]])

        g = np.linalg.solve(X.T @ X, X.T @ y)[0][0]
        gg.append(g)


    x = np.linspace(-1, 1)
    ghat = np.mean(gg)
    print("ghat", ghat)

    pylab.plot(x, f(x))
    pylab.plot(x, ghat*x, label = "mean g")
    pylab.legend()
    pylab.show()
    return ghat

def runQ5_6():
    errors = []
    kk = []
    varss = []
    ghat = runQ4_2() 
    for i in range(10000):
        x1 = rd.uniform(-1,1)
        x2 = rd.uniform(-1,1)
        X = np.array([[x1], [x2]])
    
        y1 = f(x1)
        y2 = f(x2)
        y = np.array([[y1],[y2]])

        k = np.linalg.solve(X.T @ X, X.T @ y)[0][0]
        var = scipy.integrate.quad(lambda x: (k*x-ghat*x)**2, -1, 1)
        error = scipy.integrate.quad(lambda x: (k*x - f(x))**2, -1, 1)

        kk.append(k)
        varss.append(var)
        errors.append(error)

    variance = np.mean(varss)
    error = np.mean(errors)
    bias = error - variance
    print("variance", variance)
    print("error", error)
    print("bias", bias)
       
#runQ4_2()
#runQ5_6()


#################################################









