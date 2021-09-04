# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 19:27:52 2021

@author: Mahmu
"""

import numpy
import random 
import pylab
import imp

def plot(samplePoints, weights = None, x1 = None, y1 = None, x2 = None, y2 = None):
    red_x = []
    red_y = []
    blue_x = []
    blue_y = []

    for point in samplePoints:
        if point[3] == -1.0:
            red_x.append(point[1])
            red_y.append(point[2])
        else:
            blue_x.append(point[1])
            blue_y.append(point[2])

    pylab.plot(red_x, red_y, 'ro', label = '-1\'s')
    pylab.plot(blue_x, blue_y , 'bo', label = '1\'s')
    x = numpy.array( [-1,1] )
    if x1 is not None:
        # plot target function(black) and hypothesis function(red) lines
        slope = (y2-y1)/(x2-x1)
        intercept = y2 - slope * x2
        pylab.plot(x, slope*x + intercept, 'r')
    if weights is not None:
        pylab.plot( x, -weights[1]/weights[2] * x - weights[0] / weights[2] , linewidth = 2, c ='g', label = 'g') # this will throw an error if w[2] == 0
    pylab.ylim([-1,1])
    pylab.xlim([-1,1])
    pylab.legend()
    pylab.show()


"""
Calculate weights using linear regression.
Return list of weights.
"""
def linearRegression(samplePoints):
    X = []
    y = []
    y_location = len(samplePoints[0]) -1 # y's location is assumed to be the last element in the list

    # Construct X space and split y values out
    for point in samplePoints:
        X.append(numpy.array(point[:y_location]))
        y.append(point[y_location])

    X = numpy.array(X)
    y = numpy.array(y)
    X_inverse = numpy.linalg.pinv(X)
    
    return numpy.dot(X_inverse, y)

pointsL = []
# ########################################
# Perceptron helper functions from HW 1 ##
# ########################################
def generatePoints(numberOfPoints):
##    random.seed(1) #used for testing
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)
    points = []

    
    for i in range (numberOfPoints):
##        random.seed(1)
        x = random.uniform (-1, 1)
        y = random.uniform (-1, 1)
        points.append([1, x, y, hw1TargetFunction(x1, y1, x2, y2, x, y)]) # add 1/-1 indicator to the end of each point list
    return x1, y1, x2, y2, points

def hw1TargetFunction(x1,y1,x2,y2,x3,y3):
    u = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    if u >= 0:
        return 1
    elif u < 0:
        return -1

# ##########################################

"""
Returns E_in error percentage for given weights and sample points.
Assumes samplePoints is a list of lists, and the last element in given list
is the y value.
"""
def Ein(weights, samplePoints):
    errorCount = 0
    y_location = len(samplePoints[0]) - 1
    for point in samplePoints:
        if numpy.sign(numpy.dot(weights,point[:y_location])) != point[y_location]:
            errorCount += 1

    return errorCount/float(len(samplePoints))


"""
Calculates the average E_in error of desired number of trials, using a new
set of sample points each time.
Returns average in sample error.
"""
def runQ5EinSimulation(numberOfTrials, numberOfPoints):
    ein_results = []
    for i in range(numberOfTrials):
        x1, y1, x2, y2, points = generatePoints(numberOfPoints)
        ein_results.append(Ein(linearRegression(points), points))

    return numpy.mean(ein_results)

print(runQ5EinSimulation(1000, 100))
#x1,y1,x2,y2,samplePoints = generatePoints(5)
#plot(samplePoints, linearRegression(samplePoints), x1, y1, x2, y2)