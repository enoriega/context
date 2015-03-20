'''Misc functions used in general '''
from __future__ import division
import math
import itertools
import numpy as np

def bin_vec(n, max=32):
    '''Returns a list with the binary representation of n in BigEndian format'''

    return [int(i) for i in reversed(list(('{0:%ib}' % math.log(max, 2)).format(n).replace(' ', '0')))]


def eval(gold, predicted):
    normalizer = float(gold.shape[0] * gold.shape[1])
    score = np.logical_xor(predicted, gold).sum() / normalizer

    components = np.logical_xor(predicted, gold).sum(axis=1) / float(gold.shape[1])

    return score, list(itertools.chain(*components.tolist()))


def split_data(data, slices=1):
    ''' Data should be a numpy array or matrix'''

    if slices == 0:
        return [data]

    step = int(data.shape[0] / slices)

    indices = [0]
    prev = 0
    for i in xrange(slices-1):
        ix = prev + step
        indices.append(ix)
        prev = ix

    indices.append(data.shape[0])

    ret = []
    for i in xrange(slices):
        ret.append(data[indices[i]:indices[i+1], :])

    return ret
