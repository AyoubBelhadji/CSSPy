
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


class projection_DPP_Sampler:
    def __init__(self, A, k, Q,N):
        self.A = A
        self.Q = Q
        self.N = N
        self.Q_temp = deepcopy(Q)
        self.k = k
        self.sampling_list = []
        self.column_selected_temp = np.zeros(k)
        self.sampling_round = 0
        self.lvs_array = self.Estimate_Leverage_Scores()
    def OneRound(self):
        sampled_indices = np.random.choice(self.N, 1, replace=True, p=list(self.lvs_array))
        #self.sampling_list.append(sampled_indices)
        column_selected = self.Q_temp[:,sampled_indices[0]]
        #print(np.linalg.norm(column_selected))
        self.column_selected_temp = 1/np.linalg.norm(column_selected)*column_selected
        #print(np.linalg.norm(self.column_selected_temp))
        self.Project_On_The_Vector_Orthogonal()
        self.sampling_round = self.sampling_round +1
        return sampled_indices[0]
    def MultiRounds(self):
        self.Q_temp = deepcopy(self.Q)
        self.sampling_list = []
        self.column_selected_temp = np.zeros(self.k)
        self.sampling_round = 0      
        for t in range(self.k):
            self.lvs_array = self.Estimate_Leverage_Scores()  
            sampled_indices_ = self.OneRound()
            self.sampling_list.append(sampled_indices_)
        return self.A[:,self.sampling_list]
    def Estimate_Leverage_Scores(self):
        return 1/(self.k-self.sampling_round)*np.diag(np.dot(self.Q_temp.T,self.Q_temp))
    def Evaluate_Approximation_error_fro(self,A_S):
        approximation_error_function_fro(self.k,self.A,A_S)
        return approximation_error_function_fro(self.k,self.A,A_S)

    def Project_On_The_Vector_Orthogonal(self): 
        #print(self.column_selected_temp)
        projection_matrix = np.eye(self.k-self.sampling_round) - np.outer(np.transpose(self.column_selected_temp),self.column_selected_temp)
        #print(projection_matrix)
        #print(projection_matrix_2)
        lambda_, W = np.linalg.eigh(projection_matrix)
        #print(np.shape(self.Q_temp))
        #print(np.linalg.norm(self.column_selected_temp))
        self.Q_temp = self.Q_temp - np.outer(self.column_selected_temp,np.dot(self.column_selected_temp,self.Q_temp))
        self.Q_temp = np.dot(W.T,self.Q_temp)
        self.Q_temp = np.delete(self.Q_temp, 0, axis=0)
        #qr_q_test,_ = np.linalg.qr(self.Q_temp.T)
        #self.Q_temp = qr_q_test.T
        #print(np.rank(self.Q_temp))
        #print(np.trace(np.dot(self.Q_temp,np.transpose(self.Q_temp))))

