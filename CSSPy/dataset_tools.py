import sys
#sys.path.append('../')
sys.path.insert(0, '..')

from CSSPy.evaluation_functions import *
import scipy.io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import random, linalg, dot, diag, all, allclose
import timeit
from scipy.sparse.linalg import svds


def calculate_right_eigenvectors_k_svd(X_,k):
    _,_,V_k_ = svds(X_, k,  return_singular_vectors='vh')
    
    return (V_k_)

