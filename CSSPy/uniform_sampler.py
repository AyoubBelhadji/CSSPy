
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
    """ Uniform Sampler object
    """
    def __init__(self, A, k,N):
        """ Create a Uniform Sampler for the matrix :math:`A` for k-low rank apparoximation.
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
        self.sampling_list = []
        self.sampling_round = 0      
        self.sampling_list = self.OneRound()
        return self.A[:,self.sampling_list]