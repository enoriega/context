from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
import sys
import pickle
import pyhsmm
from utils import *
from pyhsmm.util.text import progprint_xrange


from distributions import Probit

if len(sys.argv) > 1:
    sample_num = int(sys.argv[1])
else:
    sample_num = 100

threads = 4

#### load data

#with open('pyhsmm_data.pickle', 'r') as f:
with open('synt_data.pickle', 'r') as f:
    dataset = pickle.load(f)

data = dataset['obs']
W = dataset['W']
latent = dataset['states']

N = 32   # Number of states
obs_dim = data.shape[1]

#########################
#  posterior inference  #
#########################

# Set the weak limit truncation level
Nmax = N

# and some hyperparameters
obs_dim = data.shape[1]

dur_hypparams = {'alpha_0':10*1,
                 'beta_0':10*100}

obs_distns = [Probit(W, np.matrix(l)) for l in [ [int(i) for i in reversed(list('{0:5b}'.format(state).replace(' ', '0')))] for state in xrange(N)]]
dur_distns = [pyhsmm.distributions.GeometricDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha=6.,gamma=6., # these can matter; see concentration-resampling.py
        init_state_concentration=6., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)
posteriormodel.add_data(data,trunc=20) # duration truncation speeds things up when it's possible

for idx in progprint_xrange(sample_num):
    posteriormodel.resample_model()

print "Evaluating the results"

# Get the predicted states
states = np.hstack([np.matrix(l).T for l in [ [int(i) for i in reversed(list('{0:5b}'.format(state).replace(' ', '0')))] for state in posteriormodel.stateseqs[0]]])


# Compute accuracy
overall, components = eval(latent, states)

print "Overall Accuracy: %f" % overall
print
for i, c in enumerate(components):
    print "Accurary for component %i: %f" % (i+1, c)
