
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



class derandomized_Projection_DPP_Sampler:
    def __init__(self, A, k,Q,N):
        self.A = A
        self.N = N
        self.Q = deepcopy(Q)
        self.Q_temp = deepcopy(Q)
        self.k = k
        self.sampling_list = [585]
        self.avoiding_list = list(range(self.N))
        self.column_selected_temp = np.zeros(k)
        self.sampling_round = 0
        self.vs_array = self.Estimate_Scores()
    def OneRound(self):
        sampled_indices = np.argmin(self.vs_array)
        print("selected")
        print(sampled_indices)
        #self.sampling_list.append(sampled_indices)
        column_selected = self.Q_temp[:,sampled_indices]
        q_i = 1/np.linalg.norm(column_selected)*column_selected
        #b_i = column_selected
        #print(sampled_indices)
        #print(b_i)
        #print(np.linalg.norm(self.column_selected_temp))
        Q_loop_temp = deepcopy(self.Q_temp)
        self.B_temp = self.Project_The_Matrix_On_The_Vector_Orthogonal(Q_loop_temp,q_i)
        #print(self.B_temp)
        #print(self.B_temp[:,sampled_indices])
        self.sampling_round = self.sampling_round +1
        return sampled_indices
    def MultiRounds(self):
        #self.B_temp = deepcopy(self.B)
        #self.sampling_list = []
        #print(self.Estimate_Scores())
        self.column_selected_temp = np.zeros(self.k)
        self.sampling_round = 0      
        for t in range(self.k-1):
            print("step")
            print(t)
            sampled_indices = self.OneRound()
            self.sampling_list.append(sampled_indices)
            #print(self.sampling_list)
            #print(self.avoiding_list)
            #self.avoiding_list.remove(sampled_indices)           
            self.vs_array = self.Estimate_Scores()  

        return self.A[:,self.sampling_list]
    def Estimate_Scores(self):
        temp_scores = [0]*self.N
        d_Q,_ = np.shape(self.Q_temp)
        Q_loop_temp = deepcopy(self.Q_temp)
        for i in range(self.N):
            #print(i)
            if i not in self.sampling_list:
                temp_score = 0
                #print(i)
                #score_l_m = self.Calculate_Subset_Volume(column_selected_S_l_m)*self.Calculate_Subset_Covolume(self.Q,column_selected_S_l_m,self.sampling_round+1)
                #temp_score += score_l_m
                for l in self.sampling_list:
                    S_l_m = self.Swap_List_Element(self.sampling_list,l,i)
                    #print(S_l_m)

                    column_selected_S_l_m = self.Q[:,S_l_m]
                    score_l_m = self.Calculate_Subset_Volume(column_selected_S_l_m)*self.Calculate_Subset_Covolume(self.Q,column_selected_S_l_m,self.sampling_round+1)
                    temp_score += np.abs(score_l_m)

                temp_scores[i] = temp_score
            else:
                temp_scores[i] = np.inf
        return temp_scores
    def Estimate_Scores_2(self):
        temp_scores = [0]*self.N
        d_Q,_ = np.shape(self.Q_temp)
        Q_loop_temp = deepcopy(self.Q_temp)
        for i in range(self.N):
            print(i)
            if i not in self.sampling_list:
                temp_score = 0
                for l in self.sampling_list:
                    for m in self.avoiding_list:
                        S_l_m = self.Swap_List_Element(self.sampling_list,l,m)
                        column_selected_S_l_m = self.Q[:,S_l_m]
                        print(S_l_m)
                        score_l_m = self.Calculate_Subset_Volume(column_selected_S_l_m)*self.Calculate_Subset_Covolume(self.Q,column_selected_S_l_m,self.sampling_round)
                        temp_score += score_l_m

                temp_scores[i] = temp_score
            else:
                temp_scores[i] = np.inf

        return temp_scores
    def Swap_List_Element(self,l_0,a,b):
        temp_l = deepcopy(l_0)
        temp_l.remove(a)
        temp_l.append(b)
        return temp_l
    def Remove_List_Element(self,l_0,a):
        temp_l = deepcopy(l_0)
        temp_l.remove(a)
        #print(self.sampling_list)
        return temp_l
        #return np.asarray([int(1),int(2)])
    def Evaluate_Approximation_error_fro(self,A_S):
        approximation_error_function_fro(self.k,self.A,A_S)
        return approximation_error_function_fro(self.k,self.A,A_S)

    def Project_The_Matrix_On_The_Vector_Orthogonal(self,M,v): 
        v = 1/np.linalg.norm(v)*v
        M_copy = deepcopy(M)
        temp_M = M_copy - np.outer(v,np.dot(v,M_copy))
        return temp_M
    def Project_The_Matrix_On_The_Subset_Orthogonal(self,M,M_S): 
        #v = 1/np.linalg.norm(v)*v
        #print(M_S)
        M_S_temp = deepcopy(M_S)
        q_M_S,_ = np.linalg.qr(M_S_temp)
        projection_matrix = np.eye(self.k) - np.dot(q_M_S,q_M_S.T)
        M_copy = deepcopy(M)
        temp_M = np.dot(projection_matrix,M_copy)
        return temp_M
    def Calculate_Subset_Volume(self,M_S): 
        #v = 1/np.linalg.norm(v)*v
        M_S_temp = deepcopy(M_S)
        S_temp = np.dot(M_S_temp.T,M_S_temp)
        return np.linalg.det(S_temp)
    def Calculate_Subset_Covolume(self,M,M_S,degree): 
        #v = 1/np.linalg.norm(v)*v
        M_copy = deepcopy(M)
        M_S_copy = deepcopy(M_S)
        M_on_M_S_ortho = self.Project_The_Matrix_On_The_Subset_Orthogonal(M_copy,M_S_copy)
        temp_S = np.dot(M_on_M_S_ortho,M_on_M_S_ortho.T)
        lambda_S, _ = np.linalg.eigh(temp_S)
        poly_S = np.poly(lambda_S)

        return poly_S[self.k-degree]