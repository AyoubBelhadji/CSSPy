
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
        """ Create double phase sampler for the matrix :math:`A`.
        :param A: 
            Matrix :math:`A`.
        :type A: 
            array_type
        :param Q: 
            Matrix containig the k right singular vectors of :math:`A`.
        :type Q: 
            array_type
        :param s: 
            The randomized pre selection parameter.
        :type s: 
            int
        :param k: 
            The order of low rank apparoximation.
        :type k: 
            int
        :param N: 
            The dimension of subsampling (the number of columns) of A.
        :type N: 
            int
        """        
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
        sampled_indices = np.random.choice(self.N, self.phase_one_s, replace=True, p=list(self.lvs_array))
        column_selected = self.Q[:,sampled_indices]
        D_Q = np.diag(np.dot(column_selected.T,column_selected))
        self.phase_one_sampling_list = sampled_indices
        temp_Q = np.dot(column_selected,np.linalg.inv(np.diag(np.sqrt(D_Q))))
        return temp_Q
    def PhaseTwo(self,phase_one_Q):
        _, _, permutation_QR = linalg.qr(phase_one_Q, pivoting=True)      
        self.phase_two_sampling_list = list(permutation_QR)[0:self.k]

