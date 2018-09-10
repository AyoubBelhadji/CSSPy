
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



class double_Phase_Sampler:
    def __init__(self, A, k, Q,N,s):
        self.A = A

        self.Q = np.transpose(Q[0:k,:])
        self.N = N
        self.k = k
        self.phase_one_s = s
        self.phase_one_sampling_list = []
        self.phase_two_sampling_list = []
        self.sampling_list = []
        self.column_selected_temp = np.zeros(k)
        self.lvs_array = self.Estimate_Leverage_Scores()

    def Estimate_Leverage_Scores(self):
        return 1/(self.k)*np.diag(np.dot(self.Q,self.Q.T))

    def DoublePhase(self):
        phase_one_Q = self.PhaseOne()
        self.PhaseTwo(phase_one_Q)
        for t in self.phase_one_sampling_list:
            if t in self.phase_two_sampling_list:
                self.sampling_list.append(t)
        return self.A[:,sampled_indices_]
    def PhaseOne(self):
        temp_list = list(reversed(np.argsort(self.lvs_array)))
        sampled_indices_ = temp_list[0:self.k]
        self.phase_one_sampling_list = sampled_indices_
        return self.Q[:,sampled_indices_]
    def PhaseTwo(self,phase_one_Q):
        _, _, permutation_QR = linalg.qr(phase_one_Q, pivoting=True)      
        self.phase_two_sampling_list = list(permutation_QR)[0:self.k]

