# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 19:24:16 2021

@author: Mahmu
"""

import numpy as np
import random as rd
import pylab

"""Hoeffding Inequality
(A) Run a computer simulation for flipping 1,000 virtual fair coins. Flip each coin independently 10 times. 
(B) Focus on 3 coins as follows: 
    c_1 is the first coin flipped,
    c_rand is a coin chosen randomly from the 1,000,
    c_min is the coin which had the minimum frequency of heads (pick the earlier one in case of a tie).
    
    Let ν_1, ν_rand, and ν_min be the fraction of heads obtained for the 3 respective coins out of the 10 tosses.

Run the experiment 100,000 times in order to get a full distribution of ν1, νrand, and
νmin (note that crand and cmin will change from run to run)."""
    
""" Logicflow
3 coin results are needed, 2/3 are 'given' one needs to be derived.

General: flip each coin using random, each flip can be 1 or 0. 1 = heads, 0 = tails.
    
     1. c_first: first coin in the coinSet, take the mean of the amountOfRuns,
                     the first coin sum in every coinSet is added to the amountOfRuns.
     2. c_rand: Generate a random coin amongst coinSet using the 'random' package, in this case, 0-999.
                    Sum up the heads and add them to the rand_Coin_List. Take the mean, /10 and print.
     3. c_min: Max is 10/10 times. If the sum(coin) < Max, take it as the currentMin. Add to list, repeat.
                   Do this amountOfRuns times. Then take the mean, /10 and print.
"""

firstCoinSet = []
randCoinSet = []
minCoinSet = []

def generateCoinSet(amountOfCoins, amountOfFlipsPerCoin):
    listOfCoinsFlips = []
    for coin in range(amountOfCoins):
        coin = []
        for flip in range(amountOfFlipsPerCoin):
            coin.append(rd.randint(0,1))
        listOfCoinsFlips.append(coin)
    return listOfCoinsFlips

def getC_min(coinSet):
        currentMin = 10
        for coin in coinSet:
            if sum(coin) < currentMin:
                currentMin = sum(coin)
            if sum(coin) == 0:
                break    
        return currentMin
    
def printAllAndTakeMeanOfMu(firstCoinSet, randCoinSet, minCoinSet):
    print("c_first: " + str(np.mean(firstCoinSet)))
    print("c_rand: " + str(np.mean(randCoinSet)))
    print("c_min: " + str(np.mean(minCoinSet)))
    
def appendNuValuesToSets(coinSet, currentMin):
        firstCoinSet.append(sum(coinSet[0])/10.0)
        randCoinSet.append(sum(coinSet[rd.randint(0,999)])/10.0)
        minCoinSet.append(currentMin/10.0)

def simulationGetAllMu(amountOfRuns, amountOfCoins, amountOfFlips):   
    for eaRun in range(amountOfRuns):
        coinSet = generateCoinSet(amountOfCoins, amountOfFlips)
        currentMin = getC_min(coinSet)   
        appendNuValuesToSets(coinSet, currentMin)
    printAllAndTakeMeanOfMu(firstCoinSet, randCoinSet, minCoinSet)


"""2. Which coin(s) has a distribution of ν that satisfies the (single-bin) Hoeffding
Inequality?
[a] c_1 only
[b] c_rand only
[c] c_min only
[d] c_1 and crand
[e] c_min and crand
"""

def hoeffdingBound(epsilon, N):
    P = 2.0*(np.exp(-2*epsilon*epsilon*N))
    return P

trueBox = []
def inequlityAbsE(nu,mu,epsilon):
    trueCount = 1
    if abs(nu-mu) > epsilon:
        trueBox.append(trueCount)
        
def checkProbabilityOfInequility(nuList,mu,epsilon):
    for nu in nuList: 
        inequlityAbsE(nu,mu,epsilon)
    return sum(trueBox)/amountOfRuns 

def checkIfHoeffdingIsTrueMin(minCoinSet,mu,epsilon,N):
    if checkProbabilityOfInequility(minCoinSet,mu,epsilon) <= hoeffdingBound(epsilon, N):
        print("c_min holds for this Ɛ: " + str(epsilon))
    
def checkIfHoeffdingIsTrueRand(randCoinSet,mu,epsilon,N):
    if checkProbabilityOfInequility(randCoinSet,mu,epsilon) <= hoeffdingBound(epsilon, N):
        print("c_rand holds for this Ɛ: "  + str(epsilon))
        
def checkIfHoeffdingIsTrueFirst(firstCoinSet,mu,epsilon,N):
    if checkProbabilityOfInequility(firstCoinSet,mu,epsilon) <= hoeffdingBound(epsilon, N):
        print("c_first holds for this Ɛ: " + str(epsilon))        
    
def results(minCoinSet,randCoinSet,firstCoinSet,epsilon, N, amountOfRuns, amountOfCoins):

    simulationGetAllMu(amountOfRuns,amountOfCoins,N) 
    # c_first: 0.500756
    # c_rand: 0.5004669999999999
    # c_min: 0.037353000000000004   
        
    #Check for c_min:
    checkIfHoeffdingIsTrueMin(minCoinSet,np.mean(minCoinSet),epsilon,N)   

    #Check for c_rand:
    checkIfHoeffdingIsTrueRand(randCoinSet,np.mean(randCoinSet),epsilon,N)  

    #Check for c_first:
    checkIfHoeffdingIsTrueFirst(firstCoinSet,np.mean(firstCoinSet),epsilon,N) 

epsilon = 0.03
amountOfCoins = 1000
N = 10
amountOfRuns = 1000
#results(minCoinSet,randCoinSet,firstCoinSet,epsilon, N, amountOfRuns, amountOfCoins)

"""
• Error and Noise
Consider the bin model for a hypothesis h that makes an error with probability µ in
approximating a deterministic target function f (both h and f are binary functions).
If we use the same h to approximate a noismuy version of f given by:
P(y | x) =  λ         y = f(x)
            1 − λ     y != f(x)
3. What is the probability of error that h makes in approximating y? Hint: Two
wrongs can make a right!
2
[a] µ
[b] λ
[c] 1-µ
[d] (1 − λ) ∗ µ + λ ∗ (1 − µ)
[e] (1 − λ) ∗ (1 − µ) + λ ∗ µ CORRECT
"""
 
"""
What is the probability in error of h(x) approximating f(x), given that f(x) is noisy.
Note: h(x) = f(x) within probability of (1-mu), and h(x)!=f(x) within mu, since that the probaility of error,
i.e. the probability of the functions not equaling each other.
This is equivalent to computing: Pr(h(x)~=y). Consider 2 cases:
    (1) h(x)=f(x) and f(x)!=y; (1-mu)*(1-lambda)
    (2) h(x)!=f(x) and f(x)=y; (mu*lambda)
    Pr(h(x)~=y)=Pr(1)+Pr(2) = (1-mu)*(1-lambda) + (mu*lambda)
"""

# -----------------------------------------------------------------------------

"""
4. At what value of λ will the performance of h be independent of µ?
[a] 0
[b] 0.5
[c] 1/√2
[d] 1
[e] No values of λ
"""

"""
Pr(h(x) ~= f(x)) =  λ*mu + (1 − λ )(1 - mu) = 1 - mu - λ +  λmu +  λmu = 1+2*lamda*mu -mu -lamda
Independent of mu => mu should not exist, when lambda = 1/2 => 1+2(0.5)mu - mu - 1/2 = 1/2

"""

"""
• Linear Regression
In these problems, we will explore how Linear Regression for classification works. As
with the Perceptron Learning Algorithm in Homework # 1, you will create your own
target function f and data set D. Take d = 2 so you can visualize the problem, and
assume X = [−1, 1] × [−1, 1] with uniform probability of picking each x ∈ X . In
each run, choose a random line in the plane as your target function f (do this by
taking two random, uniformly distributed points in [−1, 1] × [−1, 1] and taking the
line passing through them), where one side of the line maps to +1 and the other maps
to −1. Choose the inputs xn of the data set as random points (uniformly in X ), and
evaluate the target function on each xn to get the corresponding output yn.
5. Take N = 100. Use Linear Regression to find g and evaluate Ein, the fraction of
in-sample points which got classified incorrectly. Repeat the experiment 1000
times and take the average (keep the g’s as they will be used again in Problem
6). Which of the following values is closest to the average Ein? (Closest is the
option that makes the expression |your answer −given option| closest to 0. Use
this definition of closest here and throughout.)
[a] 0
[b] 0.001
[c] 0.01
[d] 0.1
[e] 0.5

"""
# uniform probability to pick any on the x's
# in each run choose a random line in the plane as your target function
# distribute points above or below, color them
#             Find: E_in = #trueClassifiedPoints/#amountOfPoints
#             Reapeat #1000 times, take their mean.
#             => You get the E_in ~ E_out.

def makeDataSet(amountOfPoints):
    x_1 = rd.uniform(-1,1)
    y_1 = rd.uniform(-1,1)
    x_2 = rd.uniform(-1,1)
    y_2 = rd.uniform(-1,1)
    dataSet = []
    for i in range(amountOfPoints):
        x = rd.uniform(-1, 1)
        y = rd.uniform(-1, 1)
        p = [1,x, y,checkAboveAndBelowTF(x_1, y_1, x_2, y_2, x, y)]
        dataSet.append(p)
    return x_1,y_1,x_2,y_2, dataSet

def checkAboveAndBelowTF(x_1, y_1, x_2, y_2, p_x, p_y):
    u = (x_2-x_1)*(p_y-y_1) - (y_2-y_1)*(p_x-x_1)
    if u >= 0:
        return 1 # above the line
    elif u < 0:
        return -1

def plot(amountOfPoints):    
    # y = kx+m, m = y - kx = y_1-kx_1
    x_1,y_1,x_2,y_2,dataSet = makeDataSet(amountOfPoints)
    for p in dataSet:
        if(p[3] == 1):
            pylab.plot(p[1],p[2],'bo')
        else:
            pylab.plot(p[1],p[2],'go')    
    k = (y_2-y_1)/(x_2-x_1)
    m =  y_1-k*x_1
    x = np.linspace(-1,1,500)
    weights = getWeights(dataSet)    
    pylab.plot(x,(k*x)+m, linewidth = 3, c = 'r', label="Target Function")
    pylab.axis([-1, 1, -1, 1])
    pylab.title("Hw2: Linear Regression")
    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.plot( x, -weights[1]/weights[2] * x - weights[0] / weights[2] , linewidth = 3, c ='k', label = 'g') # this will throw an error if w[2] == 0# 5: N = 100. Use linear regression to find g
    pylab.legend()

def get_X_Y_matrix(dataSet):
    X = []
    y = []
    
    y_position = len(dataSet[0]) - 1
    
    for p in dataSet:
        X.append(p[:y_position])
        y.append(p[y_position])
    X = np.array(X)
    y = np.array(y)
    return X,y
    
def getWeights(dataSet):
    X,Y=get_X_Y_matrix(dataSet) 
    X_dagger = np.linalg.pinv(X)
    weights = np.matmul(X_dagger,Y)  
    return weights


def E_in(weights,points):
    #  the fraction of in-sample points which got classified incorrectly
     wrongs = 0
     y_location = len(points[0]) - 1
     for point in points:
         if np.sign(np.dot(weights,point[:y_location])) != point[y_location]:
             wrongs += 1

     E_in = wrongs/float(len(points))
     return E_in

def doExperiment(times, amountOfPoints):
    Ein_storage = []
    for i in range(times):     
            x1,y1,x2,y2,dataSet = makeDataSet(amountOfPoints)
            weights = getWeights(dataSet)
            Ein_storage.append(E_in(weights,dataSet))
    return np.mean(Ein_storage)


#plot(100)

def runQ6EoutSimulation(numberOfTrials, numberOfPoints):
    eout_results = []
    for i in range(numberOfTrials):
        errorCount = 0
        x1, y1, x2, y2, points = makeDataSet(numberOfPoints)
        weights = getWeights(points)

        for i in range(numberOfPoints):
            point = [1, rd.uniform(-1,1), rd.uniform(-1,1)]
            if np.sign(np.dot(point, weights)) != np.sign(checkAboveAndBelowTF(x1, y1, x2, y2, point[1], point[2])):
                errorCount += 1 
            # lines above are crucial, I check whether or not my generated function witht he weights,
            # i.e: x2 = -(w1/w2)x - w0/w2 will correctly seperate the points on each side like TF,
            # given random uniformly distributed points.

        eout_results.append(errorCount/float(numberOfPoints))

    return np.mean(eout_results)

#print(runQ6EoutSimulation(1000,1000))
# ----------------------------------------------------------------------------------------------
# Simple sign function
def sign(y):
    if y >= 0:
        return 1
    elif y < 0:
        return -1

# a.k.a dot product
def perceptronCalc(x, w):
    return x[0]*w[0] + x[1]*w[1] + x[2]*w[2]

def train(training_points, iterationLimit): #9,100
    w = getWeights(training_points) # initialize weights for w[0], w[1], w[2]
    learned = False
    iterations = 0 # keep track of the iteration count
    
    # This method is the primary PLA implentation. 
    # It returns True when all sample points are correctly classfied by the hypothesis.
    # Returns False if there was a misclassified point and the weight vector changed.
    def updateWeights():
        rd.shuffle(training_points) # randomize training points
        for point in training_points:
            result = sign(perceptronCalc(point,w)) # caclulate point and determine its sign.
            if point[3] != result: # does sample point's result match our calculated result?
                # Use line below to watch the perceptron's weights change
                # print(str(iterations) + " " + str(w) + " " + str(result) + " " + str(point) + " " + str(perceptronCalc(point,w)))
                
                # if not update weights by sample point's result
                w[0] += point[0]*point[3]
                w[1] += point[1]*point[3]
                w[2] += point[2]*point[3]


                return False # break out of loop and return
        return True # if the loop reaches this point all calculated points in the training points match their expected y's


    while not learned:
        iterations += 1
        noErrors = updateWeights() 
        if iterations == iterationLimit or noErrors:
            learned = True
            break

    return iterations, w

def doExperimentQ7(times, amountOfPoints):
    iterationList = []
    for i in range(times):     
            x1,y1,x2,y2,dataSet = makeDataSet(amountOfPoints)
            iterations,   = train(dataSet,100) 
            iterationList.append(iterations)
    pylab.hist(iterationList)
    pylab.title('# of iterations')
    pylab.show()
    return "Average # of iterations " + str(np.mean(iterationList))            



#print(doExperiment(1000,1000))
#print(doExperimentQ7(1000,10))
#plot(15)

def targetFunction(x1,x2):
    return np.sign(np.power(x1,2)+np.power(x2,2)-0.6)

def noisify(numberToNoisify, samplePoints):
    rd.shuffle(samplePoints) # randomize list
    out = []
    cnt = 0

    for point in samplePoints:
        if cnt < numberToNoisify:
            point[3] *= -1
        out.append(point)
        cnt += 1
    return out
    
def generateNoisyPoints(numberOfPoints):
    samplePoints = []
    for i in range(numberOfPoints):
        x1 = rd.uniform(-1,1)
        x2 = rd.uniform(-1,1)

        samplePoints.append([1, x1, x2, targetFunction(x1,x2)])

    return noisify(numberOfPoints/10, samplePoints)

def q8Simulation(numberOfTrials, numberOfPoints):
    results = []
    for i in range(numberOfTrials):
        points = generateNoisyPoints(numberOfPoints)
        weights = getWeights(points)
        results.append(E_in(weights, points))
        
    print(np.mean(results))

# q8Simulation(1000,1000)

def transform(point):
    return [1,point[1], point[2], point[1]*point[2], point[1]**2, point[2]**2, point[3]]

def transformPoints(samplePoints):
    out = []
    for point in samplePoints:
        out.append(transform(point))
    return out

def runQ9Simulation():
    g_s = {
        "a": [-1.0, -.05, .08, .13, 1.5, 1.5],
        "b": [-1.0, -.05, .08, .13, 1.5, 15.0],
        "c": [-1.0, -.05, .08, .13, 15.0, 1.5],
        "d": [-1.0, -1.5, .08, .13, .05, .05],
        "e": [-1.0, -.05, .08, 1.5, .15, .15]
        }
    points = transformPoints(generateNoisyPoints(1000))
    weights = getWeights(points)
    for key, value in g_s.items():
        errorCount = 0
        for point in points:
            if np.sign(np.dot(weights,point[:6])) != np.sign(np.dot(value,point[:6])):
                errorCount += 1
        print(key + " agreement: " + str(1 - errorCount/float(len(points))))

runQ9Simulation()

def runQ10Simulation(times, N):
    Errorlist = []
    for t in range(times):
        points = transformPoints(generateNoisyPoints(N))
        weights = getWeights(points)
        errorCnt = 0
        pSignLoc = len(points[0]) - 1 # len(p) = 7, sign at 6
        for point in points:
            if np.sign(np.dot(weights,point[:pSignLoc])) != point[pSignLoc]:
                errorCnt += 1
        E_in = errorCnt/N
        Errorlist.append(E_in)
    return print("E_out: " + str(np.mean(Errorlist)))

#runQ10Simulation(1000, 1000)

def runQ10Simulation2(times, N): # test weights with out of sample points, E_out ~= as in sample points
    Errorlist = []
    for t in range(times):
        points = transformPoints(generateNoisyPoints(N))
        weights = getWeights(points)
        points2 = transformPoints(generateNoisyPoints(N))
        errorCnt = 0
        pSignLoc = len(points[0]) - 1 # len(p) = 7, sign at 6
        for point2 in points2:
            if np.sign(np.dot(weights,point2[:pSignLoc])) != point2[pSignLoc]:
                errorCnt += 1
        E_in = errorCnt/N
        Errorlist.append(E_in)
    return print("E_out: " + str(np.mean(Errorlist)))

#runQ10Simulation2(1000, 1000)

""" Rationale: If the weights actually are good weights, in the sense of that they actully approximate
the targetfunction well, it should not matter which points we test the hypothesis with, the hypothesis
should correctly 'sign' both indata and out of data points correctly. The weights don't know the sign
ahead of time, so the only thing we need is to know the correct sign by using the target function to get it.
Then we can test the sign corrospondence with any point, be it in or out of sample, given that the weights
are good. If the weights are shitty, so will the results in either case."""