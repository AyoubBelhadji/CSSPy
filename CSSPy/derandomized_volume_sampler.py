
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




def elem_symm_poly(eig_val, k):
    """ Evaluate the elementary symmetric polynomials in the 
    eigen-values of the similarity kernel :math:`K`.
    :param eig_val: 
        Collection of eigen values of the similarity kernel :math:`K`.
    :type eig_val: 
        list
    
    :param k: 
        Maximum degree of elementary symmetric polynomial.
    :type k: 
        int
            
    :return: 
        poly(k,N) = :math:`e_k(\lambda_1, \cdots, \lambda_N)`
    :rtype: 
        array_type
    .. seealso::
        Algorithm 7 in :cite:`KuTa12`
        - :func:`k_dpp_KuTa12 <k_dpp_KuTa12>`
    """

    # Number of variables for the elementary symmetric polynomials to be evaluated
    N = len(eig_val)
    # Initialize output array
    poly = np.zeros((k+1, N+1)) 
    poly[0, :] = 1

    # Recursive evaluation
    for l in range(1, k+1):
        for n in range(1, N+1):
            poly[l, n] = poly[l, n-1] + eig_val[n-1] * poly[l-1, n-1]

    return poly



class derandomized_k_Volume_Sampling_Sampler:
    def __init__(self, A, k,N):
        self.A = A
        self.N = N
        self.B_temp = deepcopy(A)
        self.k = k
        self.sampling_list = []
        self.column_selected_temp = np.zeros(k)
        self.sampling_round = 0
        self.vs_array = self.Estimate_Scores()
    def OneRound(self):
        sampled_indices = np.argmin(self.vs_array)
        #self.sampling_list.append(sampled_indices)
        column_selected = self.B_temp[:,sampled_indices]
        b_i = 1/np.linalg.norm(column_selected)*column_selected
        #b_i = column_selected
        #print(sampled_indices)
        #print(b_i)
        #print(np.linalg.norm(self.column_selected_temp))
        B_loop_temp = deepcopy(self.B_temp)
        self.B_temp = self.Project_The_Matrix_On_The_Vector_Orthogonal(B_loop_temp,b_i)
        #print(self.B_temp)
        #print(self.B_temp[:,sampled_indices])
        self.sampling_round = self.sampling_round +1
        return sampled_indices
    def MultiRounds(self):
        #self.B_temp = deepcopy(self.B)
        self.sampling_list = []
        self.column_selected_temp = np.zeros(self.k)
        self.sampling_round = 0      
        for t in range(self.k):
            sampled_indices_ = self.OneRound()
            self.sampling_list.append(sampled_indices_)
            self.vs_array = self.Estimate_Scores()  
        return self.A[:,self.sampling_list]
    def Estimate_Scores(self):
        temp_scores = [0]*self.N
        d_B,_ = np.shape(self.B_temp)
        B_loop_temp = deepcopy(self.B_temp)
        for i in range(self.N):
            if i not in self.sampling_list:
                b_i = B_loop_temp[:,i]
                #print(i)
                #print(np.linalg.norm(b_i))
                B_on_ortho_i = self.Project_The_Matrix_On_The_Vector_Orthogonal(B_loop_temp,b_i)
                temp_S = np.dot(B_on_ortho_i,B_on_ortho_i.T)
                lambda_S, _ = np.linalg.eigh(temp_S)
                poly_S = np.poly(lambda_S)
                #print(i)
                temp_scores[i] = -poly_S[self.k-self.sampling_round+1]/poly_S[self.k-self.sampling_round]
            else:
                temp_scores[i] = np.inf
            #print(temp_scores[i])
            #print(poly_S)
        return temp_scores
    def Evaluate_Approximation_error_fro(self,A_S):
        approximation_error_function_fro(self.k,self.A,A_S)
        return approximation_error_function_fro(self.k,self.A,A_S)

    def Project_The_Matrix_On_The_Vector_Orthogonal(self,M,v): 
        #v = 1/np.linalg.norm(v)*v
        M_copy = deepcopy(M)
        temp_M = M_copy - np.outer(v,np.dot(v,M_copy))
        return temp_M

