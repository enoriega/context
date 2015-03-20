''' This script test the performance of making random precitions over the data set in average '''

from __future__ import division
import pickle
import sys
import numpy as np
from utils import *

# Number of iterations
if len(sys.argv) > 1:
    iters = int(sys.argv[1])
else:
    iters = 100

# Load data
with open('synt_data.pickle', 'r') as f:
   dataset = pickle.load(f)

states = dataset['states']

# Allocate space for results
overall = np.zeros(iters)
components = np.zeros((states.shape[0], iters))

# Repeat iter times
for i in xrange(iters):
    # Make a guess, The binomial dist reduces to a bernoulli if it is a single trail
    guess = np.random.binomial(1, 0.5, size=states.shape)
    overall[i], components[:, i] = eval(states, guess)


# Get the means
overall = overall.mean()
components = components.mean(axis=1)

print "Overall accuracy of random guess: %f" % overall
print
for i, c in enumerate(components):
   print "Accuracy for random guess of component %i: %f" % (i, c) 
