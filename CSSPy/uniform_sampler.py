
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



class Uniform_Sampler:
    def __init__(self, A, k,N):
        self.A = A
        #self.Q = Q
        self.N = N
        #self.Q_temp = deepcopy(Q)
        self.k = k
        self.sampling_list = []
        self.column_selected_temp = np.zeros(k)
        self.sampling_round = 0
    def OneRound(self):
        sampled_indices = np.random.choice(self.N, self.k, replace=True)
        #self.sampling_list.append(sampled_indices)
        return sampled_indices
    def MultiRounds(self):
        #self.Q_temp = deepcopy(self.Q)
        self.sampling_list = []
        #self.column_selected_temp = np.zeros(k)
        self.sampling_round = 0
        #self.lvs_array = self.Estimate_Leverage_Scores()        
        self.sampling_list = self.OneRound()
        return self.A[:,self.sampling_list]
    def Estimate_Leverage_Scores(self):
        return 1/(self.k)*np.diag(np.dot(self.Q.T,self.Q))
    def Evaluate_Approximation_error_fro(self,A_S):
        approximation_error_function_fro(self.k,self.A,A_S)
        return approximation_error_function_fro(self.k,self.A,A_S)