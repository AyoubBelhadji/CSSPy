
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

import CSSPy.strong_RRQR_algo4

class double_Phase_strong_Sampler:
    def __init__(self, A, k, Q,N,s):
        self.A = A

        self.Q = Q
        self.N = N
        self.k = k
        self.phase_one_s = s
        self.phase_one_sampling_list = []
        self.phase_two_sampling_list = []
        self.sampling_list = []
        self.column_selected_temp = np.zeros(k)
        self.lvs_array = self.Estimate_Leverage_Scores()

    def Estimate_Leverage_Scores(self):
        return 1/(self.k)*np.diag(np.dot(self.Q.T,self.Q))

    def DoublePhase(self):
        phase_one_Q = self.PhaseOne()
        self.PhaseTwo(phase_one_Q)
        count =0
        for t in self.phase_one_sampling_list:
            if count in self.phase_two_sampling_list:
                self.sampling_list.append(t)
            count += 1
        return self.A[:,self.sampling_list]
    def PhaseOne(self):
        temp_list = list(reversed(np.argsort(self.lvs_array)))
        sampled_indices = np.random.choice(self.N, self.phase_one_s, replace=True, p=list(self.lvs_array))
        #print(sampled_indices.dtype)
        #print(sampled_indices)
        column_selected = self.Q[:,sampled_indices]
        D_Q = np.diag(np.dot(column_selected.T,column_selected))
        self.phase_one_sampling_list = sampled_indices
        temp_Q = np.dot(column_selected,np.linalg.inv(np.diag(np.sqrt(D_Q))))
        #temp_Q = np.dot(column_selected,np.linalg.inv(np.diag(D_Q)))
        return temp_Q
    def PhaseTwo(self,phase_one_Q):
        _, _, permutation_QR = Strong_RRQR(phase_one_Q,self.k)     
        self.phase_two_sampling_list = list(permutation_QR)[0:self.k]

