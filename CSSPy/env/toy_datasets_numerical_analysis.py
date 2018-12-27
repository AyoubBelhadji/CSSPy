import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import binom
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, HTML

#sys.path.append("../")
from FrameBuilder.eigenstepsbuilder import *
from decimal import *
from copy import deepcopy
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from env.evaluation_functions import *

def extend_orthogonal_matrix(Q,d_target):
    [N,d] = np.shape(Q)
    determinant_test = 0
    failure_counter = 0
    #print("is nan for initial Q")
    #print(np.isnan(Q).any())
    while np.abs(determinant_test-1.0)>0.5:
        Q_target = np.zeros((N,d))
        Q_target = deepcopy(Q)
        delta = d_target - d
        for t in range(delta):
            Q_test = np.random.normal(0, 1, N)
            for _ in range(d+t):
                Q_test = Q_test - np.dot(Q_test,Q_target[:,_])*Q_target[:,_]
            Q_test = Q_test/np.linalg.norm(Q_test)
            Q_test = Q_test.reshape(N,1)
            Q_target = np.append(Q_target,Q_test,1)
        determinant_test = np.linalg.det(np.dot(Q_target,Q_target.T))
        #print("is nan for final Q")
        #print(np.isnan(Q_target).any())
        _,Sigma,_ = np.linalg.svd(Q_target, full_matrices=False)
        #print(Sigma)
        #print(str(np.linalg.det(np.dot(Q_target,Q_target.T))))
        #print(failure_counter)
        failure_counter = failure_counter +1
    return Q_target




def generate_leverage_scores_vector_with_dirichlet(d,k,nn_cardinal):
    getcontext().prec = 3
    mu_vector = np.float16(np.zeros((d,)))
    mu_vector_2 = np.float16(np.zeros((d,)))
    not_bounded = 1
    while(not_bounded == 1):
        mu_vector[0:nn_cardinal] = (k*np.random.dirichlet([1]*nn_cardinal, 1))[0]
        mu_vector = np.flip(np.sort(mu_vector),axis = 0)
        if max(mu_vector)<=1:
            not_bounded = 0
    for i in range(nn_cardinal):
        mu_vector_2[i] = round(mu_vector[i],3)
    numbers = [ int(1000*x) for x in mu_vector_2 ] 
    #print(numbers)
    #print(np.sum(numbers))
    numbers[nn_cardinal-1] = 1000*k-np.sum(numbers[0:nn_cardinal-1])
    numbers_to_float = 1/1000*np.asarray([float(x) for x in numbers])
    #print(np.sum(numbers_to_float))
    mu_vector_2 = k*numbers_to_float/np.sum(numbers_to_float)
    #print("this the real random generator")
    #print(mu_vector_2)
    return list(mu_vector_2)

def contruct_dataset_from_orthogonal_matrix(multi_Q,N,target_d,cov,mean,versions_number):
    multi_X = np.zeros((versions_number,N,target_d))
    if N<target_d:
        for t in range(versions_number):
            test_X = np.random.multivariate_normal(mean, cov, N)
            [U,_,_] = np.linalg.svd(test_X, full_matrices=False)
            U_good_shape = np.zeros((N,target_d))
            U_good_shape[:,0:N] = U
            Q_test = extend_orthogonal_matrix(multi_Q[t,:,:],target_d)
            multi_X[t,:,:] = np.dot(np.dot(Q_test,cov),U_good_shape.T).T
            _,Sigma,_ = np.linalg.svd(multi_X[t,:,:], full_matrices=False)
            #print(Sigma)
    else:
        for t in range(versions_number):
            test_X = np.random.multivariate_normal(mean, cov, N)
            [U,_,_] = np.linalg.svd(test_X, full_matrices=False)
            Q_test = extend_orthogonal_matrix(multi_Q[t,:,:],target_d)
            multi_X[t,:,:] = np.dot(np.dot(Q_test,cov),U.T).T
            _,Sigma,_ = np.linalg.svd(multi_X[t,:,:], full_matrices=False)
            #print(Sigma)
    return multi_X

def generate_orthonormal_matrix_with_leverage_scores_ES(N,d,lv_scores_vector,versions_number,nn_cardinal_list):

    lambda_vector = np.zeros((N))
    lambda_vector[0:d] = np.ones((d))

    #mu_vector = np.linspace(1, 0.1, num=N)
    #sum_mu_vector = np.sum(mu_vector)
    #mu_vector = d/sum_mu_vector*mu_vector
    Q = np.zeros((N,d))
    previous_Q = np.zeros((versions_number,N,d))
    #mu_vector = d/N*np.ones((N,1))
    E = np.zeros((N,N)) #(d,N)
    counter = 0
    for j in nn_cardinal_list:
        #print("counter")
        #print(counter)
        failure_test = 1
        while failure_test ==1:
            mu_vector = generate_leverage_scores_vector_with_dirichlet(N,d,j)
            #print(np.sum(mu_vector))
            #print(mu_vector)
            E_test = get_eigensteps_random(mu_vector,lambda_vector,N,d)
            E_ = np.zeros((d,N+1))
            for i in range(d):
                E_[i,1:N+1] = E_test[i,:] 
            F_test = get_F(d,N,np.asmatrix(E_),mu_vector)
            if np.isnan(F_test).any() == False:
                failure_test = 0
        previous_Q[counter,:,:] = np.transpose(F_test)
        Q = np.transpose(F_test)
        counter = counter +1
    return Q,previous_Q


class Numrerical_Analysis_DPP: 
    def __init__(self,N,real_dim,r,k,versions_number,mean,cov,lv_scores,cardinal_list):
        self.N = N
        self.real_dim = real_dim
        self.r = r
        self.k = k
        self.versions_number = versions_number
        self.mean = mean
        self.cov = cov
        self.lv_scores = lv_scores
        self.cardinality_list = cardinal_list
        self.Q = np.zeros((real_dim,k))
        self.multi_Q = np.zeros((self.versions_number,real_dim,k))
        self.X = np.zeros((N,real_dim))
        self.multi_X = np.zeros((self.versions_number,N,real_dim))
        #[self.Q,self.multi_Q] = generate_orthonormal_matrix_with_leverage_scores(real_dim,k,lv_scores,versions_number,'identity')
        [self.Q,self.multi_Q] = generate_orthonormal_matrix_with_leverage_scores_ES(self.real_dim,self.k,[],self.versions_number,self.cardinality_list)
        self.multi_X = contruct_dataset_from_orthogonal_matrix(self.multi_Q,self.N,self.real_dim,self.cov,self.mean,self.versions_number)
    def contruct_dataset_from_orthogonal_matrix_4(self,multi_Q,N,target_d,cov,mean,versions_number):
        test_multi_X = np.zeros((self.versions_number,N,real_dim))
        for t in range(self.versions_number):
            test_X = np.random.multivariate_normal(mean, cov, N)
            [U,_,_] = np.linalg.svd(test_X, full_matrices=False)
            Q_test = extend_orthogonal_matrix(self.multi_Q[t,:,:],target_d)
            test_multi_X[t,:,:] = np.dot(np.dot(Q_test,cov),U.T).T
        return test_multi_X
    def get_expected_error_fro_for_volume_sampling(self):
        ## Calculate the expected error ratio for the Volume Sampling distribution for every dataset
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            res_list[t] = expected_approximation_error_fro_for_sampling_scheme(test_X,test_X,self.k,self.real_dim)
        return res_list
    def get_tight_upper_bound_error_fro_for_projection_DPP(self):
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            test_U = self.multi_Q[t,:,:].T
            res_list[t] = tight_approximation_error_fro_for_sampling_scheme(test_X,test_U,self.k,self.real_dim)
        return res_list
    def get_max_diag_sum_T_matrices(self):
        res_list = np.zeros((self.versions_number))
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            _,_,test_V = np.linalg.svd(test_X, full_matrices=False)
            test_V_k = test_V[0:self.k,:]
            test_V_d_k = test_V[self.k:self.real_dim,:]
            res_list[t] = 1+np.max(np.diag(get_the_matrix_sum_T_S(self.k,self.real_dim,test_V_k,test_V_d_k)))
        return res_list
    def get_max_spectrum_sum_T_matrices(self):
        res_list = np.zeros((self.versions_number))
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            _,_,test_V = np.linalg.svd(test_X, full_matrices=False)
            test_V_k = test_V[0:self.k,:]
            test_V_d_k = test_V[self.k:self.real_dim,:]
            res_list[t] = 1+np.max(np.diag(get_the_matrix_sum_T_S(self.k,self.real_dim,test_V_k,test_V_d_k)))
        return res_list
    def get_expected_error_fro_for_projection_DPP(self):
        ## Calculate the expected error ratio for the Projection DPP distribution for every dataset
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            test_U = self.multi_Q[t,:,:].T
            res_list[t] = expected_approximation_error_fro_for_sampling_scheme(test_X,test_U,self.k,self.real_dim)
        return res_list   
    def get_expected_error_spectral_for_volume_sampling(self):
        ## Calculate the expected error ratio for the Volume Sampling distribution for every dataset
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            res_list[t] = expected_approximation_error_spectral_for_sampling_scheme(test_X,test_X,self.k,self.real_dim)
        return res_list
    def get_p_eff(self,theta=2.0):
        ## A function that calculate the p_eff.
        ## It is a measure of the concentration of V_k. This is done for every dataset
        #if theta is None:
        #    theta = 2.0
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            diag_Q_t = np.diag(np.dot(self.multi_Q[t,:,:],self.multi_Q[t,:,:].T))
            #diag_Q_t = list(diag_Q_t[::-1].sort())
            diag_Q_t = list(np.sort(diag_Q_t)[::-1])
            p = self.real_dim
            #print(diag_Q_t)
            while np.sum(diag_Q_t[0:p-1]) > float(self.k-1+1.0/theta):
                p = p-1
            res_list[t] = p
        return res_list
    def get_effective_kernel_from_orthogonal_matrix(self,theta=2.0):
        test_eff_V = np.zeros((self.versions_number,self.real_dim,self.k))
        #if theta is None:
        #    theta = 2.0
        p_eff_list = self.get_p_eff(theta)
        for t in range(self.versions_number):
            test_V = deepcopy(self.multi_Q[t,:,:])
            p_eff = p_eff_list[t]
            diag_Q_t = np.diag(np.dot(test_V,test_V.T))
            #diag_Q_t = list(diag_Q_t[::-1].sort())
            #print(diag_Q_t)
            permutation_t = list(reversed(np.argsort(diag_Q_t)))
            #print(permutation_t)
            for i in range(self.real_dim):
                if i >p_eff-1:
                    test_V[permutation_t[i],:] = 0   
            #Q_test = extend_orthogonal_matrix(self.multi_Q[t,:,:],target_d)
            test_eff_V[t,:,:] = test_V
        return test_eff_V
    def get_avoiding_probability(self):
        ## Calculate the expected error ratio for the Volume Sampling distribution for every dataset
        theta_list = [1.5,2,2.5,3,3.5,4,4.5,5]
        thetas_number = len(theta_list)
        res_list = []
        theta_count = 0
        for theta in theta_list:
            test_eff_V = self.get_effective_kernel_from_orthogonal_matrix(theta)
            for t in range(self.versions_number):
                test_U = test_eff_V[t,:,:].T
                res_list.append(np.linalg.det(np.dot(test_U,np.transpose(test_U))))
            theta_count = theta_count +1
        theoretical_list =  [1/x for x in theta_list]
        return res_list, theta_list, theoretical_list 
    def get_expected_error_fro_for_effective_kernel_sampling(self):
        ## Calculate the expected error ratio for the Volume Sampling distribution for every dataset
        res_list = np.zeros(self.versions_number)
        test_eff_V = self.get_effective_kernel_from_orthogonal_matrix()
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            test_U = test_eff_V[t,:,:].T
            res_list[t] = expected_approximation_error_fro_for_sampling_scheme(test_X,test_U,self.k,self.real_dim)
        return res_list
    def get_expected_error_spectral_for_projection_DPP(self):
        ## Calculate the expected error ratio for the Projection DPP distribution for every dataset
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            test_U = self.multi_Q[t,:,:].T
            res_list[t] = expected_approximation_error_spectral_for_sampling_scheme(test_X,test_U,self.k,self.real_dim)
        return res_list  
    def get_upper_bound_error_for_projection_DPP(self):
        ## Calculate the expected error ratio for the Projection DPP distribution for every dataset
        #res_list = np.zeros(self.versions_number+1)
        res_list = []
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            test_U = self.multi_Q[t,:,:].T
            #res_list[t] = expected_upper_bound_for_projection_DPP(test_X,test_U,self.k,self.real_dim)
            res_list.append( expected_upper_bound_for_projection_DPP(test_X,test_U,self.k,self.real_dim))
        return res_list  
    def get_error_fro_for_deterministic_selection(self):
        ## Calculate the error ratio for the k-tuple selected by the deterministic algorithm for every dataset
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            test_U = self.multi_Q[t,:,:].T
            lv_scores_vector = np.diag(np.dot(np.transpose(test_U),test_U))
            test_I_k = list(np.argsort(lv_scores_vector)[self.real_dim-self.k:self.real_dim])
            _,test_Sigma,_ = np.linalg.svd(test_X, full_matrices=False)
            res_list[t] = approximation_error_function_fro(test_Sigma,self.k,test_X,test_X[:,test_I_k])
            #res_list.append(test_I_k)
        return res_list   
    def get_error_fro_for_derandomized_projection_DPP_selection(self):
        ## Calculate the error ratio for the k-tuple selected by the deterministic algorithm for every dataset
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            test_U = self.multi_Q[t,:,:].T
            lv_scores_vector = np.diag(np.dot(np.transpose(test_U),test_U))
            test_I_k = derandomization_projection_DPP(test_U,self.k,self.real_dim)
            _,test_Sigma,_ = np.linalg.svd(test_X, full_matrices=False)
            res_list[t] = approximation_error_function_fro(test_Sigma,self.k,test_X,test_X[:,test_I_k])
            #res_list.append(test_I_k)
        return res_list 
    def get_error_spectral_for_deterministic_selection(self):
        ## Calculate the error ratio for the k-tuple selected by the deterministic algorithm for every dataset
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            test_X = self.multi_X[t,:,:]
            test_U = self.multi_Q[t,:,:].T
            lv_scores_vector = np.diag(np.dot(np.transpose(test_U),test_U))
            test_I_k = list(np.argsort(lv_scores_vector)[self.real_dim-self.k:self.real_dim])
            _,test_Sigma,_ = np.linalg.svd(test_X, full_matrices=False)
            res_list[t] = approximation_error_function_spectral(test_Sigma,self.k,test_X,test_X[:,test_I_k])
            #res_list.append(test_I_k)
        return res_list 
    def get_sum_k_leverage_scores(self):
        ## A function that calculate the k-sum: the sum of the first k k-leverage scores. It is a measure of the concentration of V_k
        ## This is done for every dataset
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            res_list[t] = estimate_sum_first_k_leverage_scores(self.multi_Q[t,:,:],self.k)
        return res_list
    def get_p_eff_leverage_scores(self):
        ## A function that calculate the k-sum: the sum of the first k k-leverage scores. It is a measure of the concentration of V_k
        ## This is done for every dataset
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            res_list[t] = get_p_eff_leverage_scores(self.multi_Q[t,:,:],self.k)
        return res_list
    def get_deterministic_upper_bound(self):
        ## A function that calculate the theoretical upper bound for the deterministic algorithm for every dataset
        res_list = np.zeros(self.versions_number)
        for t in range(self.versions_number):
            res_list[t] = 1/(1+estimate_sum_first_k_leverage_scores(self.multi_Q[t,:,:],self.k)-self.k)
        return res_list
    def get_alpha_sum_k_leverage_scores(self,alpha):
        ## A function that calculate the theoretical upper bound for the deterministic algorithm for every dataset
        res_list = np.zeros(self.versions_number)
        #k_l = self.get_sum_k_leverage_scores()
        for t in range(self.versions_number):
            k_l = estimate_leverage_scores_from_orthogonal_matrix(self.multi_Q[t,:,:])[0:k]
            func_k = np.power(np.linspace(1, k, num=k),alpha)
            res_list[t] = np.dot(func_k,k_l)
        return res_list
    def plot_scatter_distribution_vs_error(self,distr):
        if distr == "greedy":
            test_X = self.multi_X[0,:,:]
            test_U = self.multi_Q[0,:,:].T
            volumes_array_1,approximation_error_array_1 = probability_approximation_error_for_greedy_trace_distribution(test_X,test_U,self.real_dim,self.k)
            volumes_array_1 = np.divide(volumes_array_1,np.sum(volumes_array_1))
            fig = plt.figure()
            fig.suptitle('Greedy Selection', fontsize=20)
            plt.xlabel('log Error_ratio', fontsize=18)
            plt.ylabel('Trace', fontsize=16)
            vol_X_scatter = plt.scatter( np.log(approximation_error_array_1),np.log(volumes_array_1),s =3)
            plt.show()      
        if distr == "pdpp":
            test_X = self.multi_X[0,:,:]
            test_U = self.multi_Q[0,:,:].T
            volumes_array_1,approximation_error_array_1 = probability_approximation_error_for_volumoid_distribution(test_X,test_U,self.real_dim,self.k)

            fig = plt.figure()
            fig.suptitle('Projection DPP Sampling', fontsize=20)
            plt.xlabel('log Error_ratio', fontsize=18)
            plt.ylabel('log Vol', fontsize=16)
            vol_X_scatter = plt.scatter( np.log(approximation_error_array_1),np.log(volumes_array_1),s =3)
            plt.show()            
        if distr == "vols":
            test_X = self.multi_X[0,:,:]
            volumes_array_1,approximation_error_array_1 = probability_approximation_error_for_volumoid_distribution(test_X,test_X,self.real_dim,self.k)
            fig = plt.figure()
            fig.suptitle('Volume Sampling', fontsize=20)
            plt.xlabel('log Error_ratio', fontsize=18)
            plt.ylabel('log Vol', fontsize=16)
            volumes_array_1 = np.divide(volumes_array_1,np.sum(volumes_array_1))
            vol_X_scatter = plt.scatter( np.log(approximation_error_array_1),np.log(volumes_array_1),s =3)
            plt.show()  
        if distr == "effker":
            avoiding_proba_list = self.get_avoiding_probability()
            avoiding_proba_1 = avoiding_proba_list[0]
            test_X = self.multi_X[0,:,:]
            test_eff_V = self.get_effective_kernel_from_orthogonal_matrix()
            test_U = test_eff_V[0,:,:].T
            volumes_array_1,approximation_error_array_1 = probability_approximation_error_for_volumoid_distribution(test_X,test_U,self.real_dim,self.k)
            volumes_array_1 = np.divide(volumes_array_1,avoiding_proba_1)
            #null_volume_index = [i for i volumes_array_1 >0]
            fig = plt.figure()
            fig.suptitle('Effective Kernel Sampling', fontsize=20)
            plt.xlabel('log Error_ratio', fontsize=18)
            plt.ylabel('log Vol', fontsize=16)
            vol_X_scatter = plt.scatter( np.log(approximation_error_array_1),np.log(volumes_array_1),s =3)
            plt.show()  