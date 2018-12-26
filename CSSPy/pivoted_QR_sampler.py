
import scipy.io
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy import random, linalg, dot, diag, all, allclose



class Pivoted_QR_Sampler:
    """ Pivoted QR Sampler object
    """
    def __init__(self, A, k,N):
        """ Create a Pivoted QR Sampler for the matrix :math:`A` for k-low rank apparoximation.
        :param A: 
            Matrix :math:`A`.
        :type A: 
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
        self.N = N
        self.k = k
        self.sampling_list = []
        self.column_selected_temp = np.zeros(k)
        self.sampling_round = 0
    def OneRound(self):
        sampled_indices = np.random.choice(self.N, self.k, replace=True)
        return sampled_indices
    def MultiRounds(self):
        _, _, permutation_QR = linalg.qr(self.A, pivoting=True)      
        self.sampling_list = list(permutation_QR)[0:self.k]
        return self.A[:,self.sampling_list]

