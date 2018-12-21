
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



class largest_leveragescores_Sampler:
    def __init__(self, A, k, Q,N):
        self.A = A

        self.Q = np.transpose(Q[0:k,:])
        self.N = N
        self.k = k
        self.sampling_list = []
        self.column_selected_temp = np.zeros(k)
        self.lvs_array = self.Estimate_Leverage_Scores()

    def Estimate_Leverage_Scores(self):
        return 1/(self.k)*np.diag(np.dot(self.Q,self.Q.T))

    def MultiRounds(self):
        temp_list = list(reversed(np.argsort(self.lvs_array)))
        sampled_indices_ = temp_list[0:self.k]
        self.sampling_list = sampled_indices_
        return self.A[:,sampled_indices_]


