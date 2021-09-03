# -*- coding: utf-8 -*-

import numpy as np
import pylab
np.set_printoptions(suppress=True)

allData = np.loadtxt("D:\\AI\\lfd\\lfd_hw7\\in.dta.txt")
test = np.loadtxt("D:\\AI\\lfd\\lfd_hw7\\out.dta.txt")

# split in.dta: 25 for training, take away 10 for validation

# print(len(train)) 35 rows
# print(train[25:35, :]) check so you get the format you want
# print(len(train[25:35, :])) check so that they actually consitute the 10 last rows
# print(len(trainAllData[0:25, :])) check size of train set

train = allData[0:25, :]
x1_train = train[:,0]
x2_train = train[:,1]
y_train = train[:,2]

val = allData[25:35, :] # another name for validation set is dev set
x1_val = val[:,0]
x2_val = val[:,1]
y_val = val[:,2]
N_val = val.shape[0]

#  For which model is the classification error on the validation set
#  smallest?

def trans_k(x1,x2, k):
    N = x1.shape[0]
    Z = np.array([np.ones(N),x1, x2, x1**2, x2**2, x1*x2, np.abs(x1-x2), np.abs(x1+x2)]).T
    return Z[:, 0:(k+1)]

def linreg(x1,x2,y,k):
    # feature matrix Z_k
    N = x1.shape[0]
    Z_k = trans_k(x1,x2,k)
    # see lecture 3, slide 17
    Z_d = np.linalg.pinv(Z_k)
    
    # use linreg to get weights
    w = Z_d @ y
    
    return w

fi = [3,4,5,6,7]
def getWeights():
    w_tildes = []
    w_con = []
    i = 0

    for k in fi:
        w_tildes.append(linreg(x1_train, x2_train, y_train, k))
        #print(str(k) + ":", w_tildes[i])
        i += 1

    return w_tildes

def conW(w):
    for p in range(len(w)):
        w
        

# Examine k = 3 for validation matrix and weights
w = getWeights() # from training
# print(w[0])
# print("\n", trans_k(x1_val,x2_val,3))

# Classification
def predict(x1_test, x2_test, w_tilde_k):
    '''
    - Takes vectors x1_test, x2_test corresponding 
    to unseen points (x1_test, x2_test)
    - Takes hypothesis / model w_tilde_k
    - Returns predictions for these points using
    the hypothesis w_tilde_k
    '''
    
    k = w_tilde_k.shape[0] - 1
    Z_k_test = trans_k(x1_test, x2_test, k)
    return np.sign(Z_k_test @ w_tilde_k)

errs_val = []
preds_k_val = []
def checkErrs():
    i = 0
    for k in fi:
        errs_val.append(sum(y_val != predict(x1_val,x2_val,w[i]))/N_val) 
        preds_k_val.append(predict(x1_val, x2_val, w[i]))
        i += 1
    i = 0
    for k in fi:
        #print("k=", fi[i], "    => E_val =", errs_val[i])
        i += 1
        
checkErrs()

def seeDevSet():
    pylab.plot(x1_val[y_val == 1], x2_val[y_val==1], 'ro', label = '$y=+1$')
    pylab.plot(x1_val[y_val == -1], x2_val[y_val==-1], 'bo', label = '$y=-1$')
    title_string = "Validation set, $N_{val} = 10$"
    pylab.title(title_string)
    pylab.xlabel('$x_1$')
    pylab.xlabel('$x_2$')
    pylab.legend()
    pylab.xlim(-1,1)
    pylab.ylim(-1, 1)
    pylab.show()
    
# seeDevSet()
    
def seeDev4AllK():
    u = np.arange(-1.0,1.0,0.02)
    X,Y= np.meshgrid(u,u)
    
    i = 0
    for k in fi:
        fig = pylab.figure(k, dpi = 80)
        
        # plot points
        pylab.plot(x1_val[y_val == 1], x2_val[y_val==1], 'ro', label = '$y=+1$')
        pylab.plot(x1_val[y_val == -1], x2_val[y_val==-1], 'bo', label = '$y=-1$')
         
        # plot correctly classified as blue and missclassified as red     
        missclassified = (y_val != preds_k_val[i])
        pylab.plot(x1_val[missclassified], x2_val[missclassified], 'mo', 
                   label = 'missclassified')
        
        # plot decision boundary
        print(w)
        boundary = lambda x1, x2, w: w[0]*1 + w[1]*x1 + w[2]*x2 + w[3]*x1**2 + w[4]*x2**2 + w[5]*x1*x2 +w[6]* np.absolute(x1-x2) +w[7]* np.absolute(x1+x2)
        phi = boundary(X,Y, w)
        pylab.contour(X,Y,phi, [0.0], colors = 'g')
        
        title_string = "Classification of validation set\n$N_{train}=25, N_{val}=10," + " k={0}, $".format(str(k))
        title_string += ("$E_{val}=$" + str(errs_val[i]))
        pylab.title(title_string)
        pylab.xlabel('$x_1$')
        pylab.ylabel('$x_2$')
        pylab.legend()
        pylab.xlim(-1,1)
        pylab.ylim(-1,1)
        pylab.show()
        
        i += 1
        
seeDev4AllK()

""""PROBLEM: You need a 8x8 grid, of zeros too, to draw the boundary up.
If you do like I, you do get w[0] consisting of weights from 0-7;
which it shouldn't. w[0] should only contain all w[0] weights. Could be
fixed by a well engineered for-loop although redudadant."""
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    