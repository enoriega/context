from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
import sys
import pickle
import pyhsmm
from itertools import chain
from pyhsmm.util.text import progprint_xrange
from distributions import Probit
from utils import *

if len(sys.argv) > 1:
    sample_num = int(sys.argv[1])
else:
    sample_num = 100

threads = 0

#### load data

#with open('pyhsmm_data.pickle', 'r') as f:
with open('synt_data.pickle', 'r') as f:
    dataset = pickle.load(f)

data = dataset['obs']
W = dataset['W']
latent = dataset['states']

N = 32   # Number of states
obs_dim = data.shape[1]

# An emission distribution for each possible state
obs_distns = [Probit(W, np.matrix(l)) for l in [ bin_vec(state) for state in xrange(N)]]

# Build the HMM model that will represent the fitmodel
fitmodel = pyhsmm.models.HMM(
        alpha=100.,init_state_concentration=50., # these are only used for initialization
        obs_distns=obs_distns)

[fitmodel.add_data(d) for d in split_data(data, threads)] 

print 'Gibbs sampling for initialization'

for idx in progprint_xrange(sample_num):
    if threads > 0:
        fitmodel.resample_model(num_procs=threads)
    else:
        fitmodel.resample_model()

print "Evaluating the results"

# Get the predicted states
states = np.hstack([np.matrix(l).T for l in [ [int(i) for i in reversed(list('{0:5b}'.format(state).replace(' ', '0')))] for state in chain(*[seq for seq in fitmodel.stateseqs])  ]])


# Compute accuracy
overall, components = eval(latent, states)

print "Overall Accuracy: %f" % overall
print
for i, c in enumerate(components):
    print "Accurary for component %i: %f" % (i+1, c)
