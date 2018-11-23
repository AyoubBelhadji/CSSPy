
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


def Strong_RRQR(X,k):
    [N,d] = np.shape(X)
    I_N = np.eye(N)
    for t in list(range(k)):
        X_t = X[:,t]
        alpha = -cmp(X_t[t],0) * norm(X_t)
        E_t = I_N[:,t]
        U = X_t - alpha*E_t
        V = U/norm(U)
        
