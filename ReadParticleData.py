#  Demonstration Python module for Week 3
#  Scientific Computing, Fall 2021, goodman@cims.nyu.edu
#  Read a source and target points and target points from files.
 
import numpy   as np      # general numpy

SourcePointsFile = "SourcePoints"
TargetPointsFile = "TargetPoints"

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
