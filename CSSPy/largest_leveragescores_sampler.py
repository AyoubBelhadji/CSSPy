
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
        """ Create largest k-leverage scores Sampler for the matrix :math:`A` for k-low rank apparoximation.
        :param A: 
            Matrix :math:`A`.
        :type A: 
            array_type
        :param Q: 
            Matrix containig the right singular vectors of :math:`A`.
        :type Q: 
            array_type
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
        self.Q = np.transpose(Q[0:k,:])
        self.N = N
        self.k = k
        self.sampling_list = []
        self.lvs_array = self.Estimate_Leverage_Scores()
    def Estimate_Leverage_Scores(self):
        return 1/(self.k)*np.diag(np.dot(self.Q,self.Q.T))
    def MultiRounds(self):
        temp_list = list(reversed(np.argsort(self.lvs_array)))
        sampled_indices_ = temp_list[0:self.k]
        self.sampling_list = sampled_indices_
        return self.A[:,sampled_indices_]