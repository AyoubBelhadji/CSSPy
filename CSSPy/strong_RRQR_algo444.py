
import scipy.io
import numpy as np
import pandas as pd

from oct2py import octave

octave.addpath('/StrongRRQR')


def Strong_RRQR(X,k):
    Q, R, p = octave.sRRQR_rank(X,2,k)
    return p[0:k]
        
