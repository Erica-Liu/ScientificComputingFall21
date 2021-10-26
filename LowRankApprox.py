#  Problem 3 in Homework 3
#  Scientific Computing, Fall 2021, yl8801@nyu.edu
#  Least Square Fitting

import numpy   as np      # general numpy
import time

def readData(SourcePointsFile, TargetPointsFile):

    #   Source points
    inFile       = open( SourcePointsFile, "r")  # open the source points file
    firstLine    = inFile.readline()             # the first line has the number of points
    nPoints      = int(firstLine)                # convert the number from string to int
    sourcePoints = np.zeros([nPoints,3])         # the source points array
    for p in range(nPoints):
       dataLine = inFile.readline()              # there is one point per line
       words = dataLine.split()                  # each word is a number
       x     = np.float64(words[0])              # x, y, and z coordinates
       y     = np.float64(words[1])              # convert from string to float
       z     = np.float64(words[2])
       sourcePoints[p,0] = x                     # save the numbers in the numpy array
       sourcePoints[p,1] = y
       sourcePoints[p,2] = z
    inFile.close()

    #   target points

    inFile       = open( TargetPointsFile, "r")  # open the source points file
    firstLine    = inFile.readline()             # the first line has the number of points
    nPoints      = int(firstLine)                # convert the number from string to int
    targetPoints = np.zeros([nPoints,3])         # the source points array
    for p in range(nPoints):
       dataLine = inFile.readline()              # there is one point per line
       words = dataLine.split()                  # each word is a number
       x     = np.float64(words[0])              # x, y, and z coordinates
       y     = np.float64(words[1])              # convert from string to float
       z     = np.float64(words[2])
       targetPoints[p,0] = x                     # save the numbers in the numpy array
       targetPoints[p,1] = y
       targetPoints[p,2] = z
    inFile.close()
    return sourcePoints, targetPoints

def distanceSquare(x0,x1,x2, y0, y1,y2):
    return (x0-y0)**2 + (x1-y1)**2 + (x2-y2)**2

# generate matrix A
def genMatrix(sourcePoints, targetPoints):
    n_s = sourcePoints.shape[0]
    n_t = targetPoints.shape[0]
    A = np.zeros((n_t, n_s))
    for j in range(n_t): # target i
        for i in range(n_s): # source i
            A[j,i] = 1/distanceSquare(sourcePoints[j,0], sourcePoints[j,1], sourcePoints[j,2], targetPoints[i,0], targetPoints[i,1], targetPoints[i,2])
    return A

def lowRankApproximate(A, epsilon):
    pass



if __name__ == '__main__':
    SourcePointsFile = "SourcePoints.txt"
    TargetPointsFile = "TargetPoints.txt"
    startT = time.time()
    sourcePoints, targetPoints = readData(SourcePointsFile, TargetPointsFile)
    readDataT = time.time()
    print("Reading data completed, with time {time}".format(time=(readDataT - startT)))


