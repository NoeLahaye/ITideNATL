""" A collection of useful functions in xarray and/or numpy
"""

import numpy as np

def decay_exp(x, L=1., a=1.):
    """a * exp(-x/L)"""
    return a*np.exp(-x/L)
