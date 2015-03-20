''' This file generates a dataset to be used and evaluated by the models '''
from __future__ import division
import numpy as np
import scipy as sp
import utils
import pickle
from itertools import *
from distributions import Probit

D = 32
period = 150
reps =100

# Generate a transition matrix

sigma = np.zeros((32, 32))

for i in xrange(D):
    sigma[i, i] = 90

    if i-1 >= 0:
        sigma[i, i-1] = 70
    else:
        sigma[i,i] += 70

    if i-2 >= 0:
        sigma[i, i-2] = 50
    else:
        sigma[i,i] += 50

    if i-3 >= 0:
        sigma[i, i-3] = 30
    else:
        sigma[i,i] += 30

    if i + 1 < D:
        sigma[i, i+1] = 70
    else:
        sigma[i,i] += 70

    if i + 2 < D:
        sigma[i, i+2] = 50
    else:
        sigma[i,i] += 50

    if i + 3 < D:
        sigma[i, i+3] = 30
    else:
        sigma[i,i] += 30

# Normalize it

sigma /= sigma.sum(axis=1).astype(np.float)

current = np.random.randint(0, D)

seq = [current]
for i in xrange(period-1):
   state = np.random.choice(range(D), p=sigma[current, :])
   seq.append(state)
   current = state

chain = chain(*repeat(seq, reps))

bin_vectors = [utils.bin_vec(state) for state in chain]

latent_states = np.matrix(bin_vectors)
latent_states = np.transpose(latent_states)

# W weight matrix
W = np.matrix([[5,2,2,0,0], [1,4,2,1, 0], [1,1,4,1,1], [0,1,2,4,2]])

# Normalize W
W = W / W.sum(axis=1).astype(np.float)

# Center it
W = W - W.mean(axis=1)


# Generate observations
obs = [Probit(W, latent_states[:, i]).rvs().tolist()[0] for i in xrange(latent_states.shape[1])]

observations = np.matrix(obs)

# Save it
with open('synt_data.pickle', 'w') as f:
    pickle.dump({'states':latent_states, 'obs':observations, 'W':W}, f)


