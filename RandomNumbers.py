#  Demonstration Python module for Week 3
#  Scientific Computing, Fall 2021, goodman@cims.nyu.edu
#  A simple example generating random numbers
 
import numpy   as np      # general numpy
import random  as rn

rn.seed(17)            # set a "seed" so you get the same random sequence each run
mu = 2.                # the mean of Gaussian random variables
sig = 3.               # the standard deviation, variance = sig^2.

X = rn.normalvariate( mu, sig)        # get a Gaussian, mean mu, standard deviation sigma
print("Here is a random number: {X:8.3f}".format(X=X))

n = 5      # how many more independent Gaussians to print
print("Here are " + str(n) + " more, all different")  #  the number n is not "hard wired"
for i in range(n):                                    #  this is the same n
   X = rn.normalvariate( mu, sig)
   print("      {X:8.3f}".format(X=X))


