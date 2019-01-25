import sys
#sys.path.append('../')
sys.path.insert(0, '..')
from CSSPy.dataset_tools import *
from CSSPy.doublephase_sampler import *
from CSSPy.volume_sampler import *
from CSSPy.optimized_projection_dpp_sampler import *
from CSSPy.projection_dpp_sampler import *
from CSSPy.uniform_sampler import *
from CSSPy.largest_leveragescores_sampler import *
from CSSPy.pivoted_QR_sampler import *
from CSSPy.evaluation_functions import *
import scipy.io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy
import seaborn as sns
from scipy import random, linalg, dot, diag, all, allclose
import timeit
from scipy.sparse.linalg import svds

import progressbar
from time import sleep


def launch_exp_derandomization_projection_dpp(X_matrix,dataset_name,k,exp_number):

    #print(timeit.default_timer())
    _,D,V = np.linalg.svd(X_matrix)
    V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = derandomized_Projection_DPP_Sampler(X_matrix, k, V_k,d)
    A_S = NAL.MultiRounds()
    error_fro_list = [0]*exp_number
    error_spectral_list = [0]*exp_number
    delta_time_list = [0]*exp_number
    for t in range(exp_number):
        NAL = derandomized_Projection_DPP_Sampler(X_matrix, k, V_k,d)
        delta_time_list[t] = timeit.default_timer()
        print("Projection DPP, sample number")
        print(t)
        A_S = NAL.MultiRounds()
        print(NAL.sampling_list)
        while(np.linalg.matrix_rank(A_S)<k):
            print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.MultiRounds()
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            delta_time_list[t] = timeit.default_timer() - delta_time_list[t] 
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list[t]=error_temp_fro
            error_spectral_list[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            #print(error_temp_fro)
    return error_fro_list


def launch_exp_double_phase_sampler(X_matrix,dataset_name,k,exp_number,V,D,V_k,norm):
    print("Double Phase sampling")
    print("\n")
    #print(timeit.default_timer())
    #_,D,V = np.linalg.svd(X_matrix)
    #V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = double_Phase_Sampler(X_matrix, k, V_k,d,10*k)
    A_S = NAL.DoublePhase()
    #print(A_S)
    error_fro_list = [0]*exp_number
    error_spectral_list = [0]*exp_number
    delta_time_list = [0]*exp_number
    bar = progressbar.ProgressBar(maxval=5, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bar_counter = 0
    bar_update_counter = 0
    exp_number_by_five = int(exp_number/5)
    for t in range(exp_number):
        NAL = double_Phase_Sampler(X_matrix, k, V_k,d,10*k)
        delta_time_list[t] = timeit.default_timer()
        #
        #print(t)
        A_S = NAL.DoublePhase()
        #print(NAL.sampling_list)
        while(np.linalg.matrix_rank(A_S)<k):
            print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.DoublePhase()
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            bar_counter = bar_counter +1
            delta_time_list[t] = timeit.default_timer() - delta_time_list[t] 
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list[t]=error_temp_fro
            error_spectral_list[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            if bar_counter == exp_number_by_five:
                bar_update_counter = bar_update_counter +1
                bar.update(bar_update_counter)
                sleep(0.1)
                bar_counter = 0
    bar.finish()
    bar_update_counter = 0
    if norm =="fro":
        return error_fro_list
    if norm =="spectral":
        return error_spectral_list



def launch_exp_double_phase_strong_sampler(X_matrix,dataset_name,k,exp_number,V,D,V_k,norm):
    print("Double Phase strong sampling")
    print("\n")
    #print(timeit.default_timer())
    #_,D,V = np.linalg.svd(X_matrix)
    #V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = double_Phase_strong_Sampler(X_matrix, k, V_k,d,20*k)
    A_S = NAL.DoublePhase()
    #print(A_S)
    error_fro_list = [0]*exp_number
    error_spectral_list = [0]*exp_number
    delta_time_list = [0]*exp_number
    bar = progressbar.ProgressBar(maxval=5, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bar_counter = 0
    bar_update_counter = 0
    exp_number_by_five = int(exp_number/5)
    for t in range(exp_number):
        NAL = double_Phase_strong_Sampler(X_matrix, k, V_k,d,20*k)
        delta_time_list[t] = timeit.default_timer()
        #
        #print(t)
        A_S = NAL.DoublePhase()
        #print(NAL.sampling_list)
        while(np.linalg.matrix_rank(A_S)<k):
            print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.DoublePhase()
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            bar_counter = bar_counter +1
            delta_time_list[t] = timeit.default_timer() - delta_time_list[t] 
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list[t]=error_temp_fro
            error_spectral_list[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            if bar_counter == exp_number_by_five:
                bar_update_counter = bar_update_counter +1
                bar.update(bar_update_counter)
                sleep(0.1)
                bar_counter = 0
    bar.finish()
    bar_update_counter = 0
    if norm =="fro":
        return error_fro_list
    if norm =="spectral":
        return error_spectral_list







def launch_exp_projection_dpp(X_matrix,dataset_name,k,exp_number,V,D,V_k,norm):
    print("Projection DPP sampling")

    #print(timeit.default_timer())
    #_,D,V = np.linalg.svd(X_matrix)
    #V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = projection_DPP_Sampler(X_matrix, k, V_k,d)
    A_S = NAL.MultiRounds()
    error_fro_list_projection_DPP = [0]*exp_number
    error_spectral_list_projection_DPP = [0]*exp_number
    delta_time_list_projection_DPP = [0]*exp_number
    bar = progressbar.ProgressBar(maxval=5, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bar_counter = 0
    bar_update_counter = 0
    exp_number_by_five = int(exp_number/5)
    for t in range(exp_number):
        delta_time_list_projection_DPP[t] = timeit.default_timer()
        A_S = NAL.MultiRounds()
        while(np.linalg.matrix_rank(A_S)<k):
            print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.MultiRounds()
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            delta_time_list_projection_DPP[t] = timeit.default_timer() - delta_time_list_projection_DPP[t] 
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list_projection_DPP[t]=error_temp_fro
            error_spectral_list_projection_DPP[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            if bar_counter == exp_number_by_five:
                bar_update_counter = bar_update_counter +1
                bar.update(bar_update_counter)
                sleep(0.1)
                bar_counter = 0
    bar.finish()
    bar_update_counter = 0
            #print(error_temp_fro)

    if norm =="fro":
        return error_fro_list_projection_DPP
    if norm =="spectral":
        return error_spectral_list_projection_DPP



def launch_exp_optimized_projection_dpp(X_matrix,dataset_name,k,exp_number,V,D,V_k,norm):
    print("Optimized Projection DPP sampling")
    #print(timeit.default_timer())

    #_,D,V = np.linalg.svd(X_matrix)
    #V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = optimized_projection_DPP_Sampler(X_matrix, k,D, V,d)
    A_S = NAL.MultiRounds()
    error_fro_list_optimized_projection_DPP = [0]*exp_number
    error_spectral_list_optimized_projection_DPP = [0]*exp_number
    delta_time_list_optimized_projection_DPP = [0]*exp_number
    bar = progressbar.ProgressBar(maxval=5, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bar_counter = 0
    bar_update_counter = 0
    exp_number_by_five = int(exp_number/5)
    for t in range(exp_number):
        delta_time_list_optimized_projection_DPP[t] = timeit.default_timer()
        A_S = NAL.MultiRounds()
        while(np.linalg.matrix_rank(A_S)<k):
            print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.MultiRounds()
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            delta_time_list_optimized_projection_DPP[t] = timeit.default_timer() - delta_time_list_optimized_projection_DPP[t] 
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list_optimized_projection_DPP[t]=error_temp_fro
            error_spectral_list_optimized_projection_DPP[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            #print(error_temp_fro)
            if bar_counter == exp_number_by_five:
                bar_update_counter = bar_update_counter +1
                bar.update(bar_update_counter)
                sleep(0.1)
                bar_counter = 0
    bar.finish()
    bar_update_counter = 0
    if norm =="fro":
        return error_fro_list_optimized_projection_DPP
    if norm =="spectral":
        return error_spectral_list_optimized_projection_DPP

def get_p_eff_leverage_scores(Q,k,theta):
    Q_ = np.transpose(Q) 
    lv_scores_vector = estimate_leverage_scores_from_orthogonal_matrix(Q_)
    p = 0
    lv_scores_sum_until_p = 0
    while lv_scores_sum_until_p < k-1+1/theta:
        lv_scores_sum_until_p = lv_scores_sum_until_p+lv_scores_vector[p]
        p = p +1
    return p

def estimate_leverage_scores_from_orthogonal_matrix(Q):
    [N,_] = np.shape(Q)
    lv_scores_vector = np.zeros((N,1))
    lv_scores_vector = np.diag(np.dot(Q,np.transpose(Q)))
    lv_scores_vector = np.asarray(list(reversed(np.sort(lv_scores_vector))))
    return lv_scores_vector

def get_forbidden_subset(Q,p_eff):
    Q_ = np.transpose(Q)
    [N,_] = np.shape(Q_)
    lv_scores_vector = np.zeros((N,1))
    lv_scores_vector = np.diag(np.dot(Q_,np.transpose(Q_)))
    I = list(reversed(list(np.argsort(lv_scores_vector))))
    #lv_scores_vector = np.asarray(list(reversed(np.sort(lv_scores_vector))))
    res_list = list(range(p_eff+1,N))
    forbidden_list = [I[i] for i in res_list]
    return forbidden_list


def get_random_forbidden_subset(Q,p_eff):
    Q_ = np.transpose(Q)
    [N,k] = np.shape(Q_)
    lv_scores_vector = np.zeros((N,1))
    lv_scores_vector = np.diag(np.dot(Q_,np.transpose(Q_)))/k
    print("here")
    print(np.sum(lv_scores_vector))
    
    #sampled_indices = np.random.choice(N, p_eff, replace=False, p=lv_scores_vector)
    sampled_indices = np.random.choice(N, p_eff, replace=False)
    #diff_list = set(list(range(N))).symmetric_difference(sampled_indices)
    diff_list = list(set(list(range(N)))- set(sampled_indices))
    print(np.sum(lv_scores_vector[sampled_indices]))
    return diff_list


def launch_exp_effective_projection_dpp(X_matrix,dataset_name,k,exp_number,theta,V,D,V_k,norm):
    print("Optimized Projection DPP sampling")
    #print(timeit.default_timer())

    #_,D,V = np.linalg.svd(X_matrix)
    #V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)

    peff_theta = get_p_eff_leverage_scores(V_k,k,theta)
    
    forbidden_subset = get_forbidden_subset(V_k,peff_theta)
    
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = optimized_projection_DPP_Sampler(X_matrix, k,D, V,d)
    A_S = NAL.MultiRounds()
    error_fro_list_optimized_projection_DPP = [0]*exp_number
    error_spectral_list_optimized_projection_DPP = [0]*exp_number
    delta_time_list_optimized_projection_DPP = [0]*exp_number
    bar = progressbar.ProgressBar(maxval=5, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bar_counter = 0
    bar_update_counter = 0
    exp_number_by_five = int(exp_number/5)
    for t in range(exp_number):
        delta_time_list_optimized_projection_DPP[t] = timeit.default_timer()
        A_S = NAL.MultiRounds()
        intersection_list = list(set(forbidden_subset) & set(NAL.sampling_list))
        print(t)
        while(np.linalg.matrix_rank(A_S)<k or len(intersection_list)>0):
            print("fail")
            print(len(forbidden_subset))
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.MultiRounds()
            intersection_list = list(set(forbidden_subset) & set(NAL.sampling_list))
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            delta_time_list_optimized_projection_DPP[t] = timeit.default_timer() - delta_time_list_optimized_projection_DPP[t] 
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list_optimized_projection_DPP[t]=error_temp_fro
            error_spectral_list_optimized_projection_DPP[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            #print(error_temp_fro)
            if bar_counter == exp_number_by_five:
                bar_update_counter = bar_update_counter +1
                bar.update(bar_update_counter)
                sleep(0.1)
                bar_counter = 0
    bar.finish()
    bar_update_counter = 0
    if norm =="fro":
        return error_fro_list_optimized_projection_DPP
    if norm =="spectral":
        return error_spectral_list_optimized_projection_DPP

def launch_exp_random_effective_projection_dpp(X_matrix,dataset_name,k,exp_number,peff_random,V,D,V_k,norm):
    print("Optimized Projection DPP sampling")
    #print(timeit.default_timer())

    #_,D,V = np.linalg.svd(X_matrix)
    #V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)

    #peff_theta = get_p_eff_leverage_scores(V_k,k,2)
    
    #forbidden_subset = get_forbidden_subset(V_k,peff_theta)

    forbidden_subset = get_random_forbidden_subset(V_k,peff_random)
    
    
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = optimized_projection_DPP_Sampler(X_matrix, k,D, V,d)
    A_S = NAL.MultiRounds()
    error_fro_list_optimized_projection_DPP = [0]*exp_number
    error_spectral_list_optimized_projection_DPP = [0]*exp_number
    delta_time_list_optimized_projection_DPP = [0]*exp_number
    bar = progressbar.ProgressBar(maxval=5, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bar_counter = 0
    bar_update_counter = 0
    exp_number_by_five = int(exp_number/5)
    for t in range(exp_number):
        print(t)
        delta_time_list_optimized_projection_DPP[t] = timeit.default_timer()
        A_S = NAL.MultiRounds()
        
        intersection_list = list(set(forbidden_subset) & set(NAL.sampling_list))
        while(np.linalg.matrix_rank(A_S)<k or len(intersection_list)>0):
            #if(len(intersection_list)>0):
            #    print(len(list(forbidden_subset)))
            #print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.MultiRounds()
            intersection_list = list(set(forbidden_subset) & set(NAL.sampling_list))
            #print(NAL.sampling_list)
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            delta_time_list_optimized_projection_DPP[t] = timeit.default_timer() - delta_time_list_optimized_projection_DPP[t] 
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list_optimized_projection_DPP[t]=error_temp_fro
            error_spectral_list_optimized_projection_DPP[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            #print(error_temp_fro)
            if bar_counter == exp_number_by_five:
                bar_update_counter = bar_update_counter +1
                bar.update(bar_update_counter)
                sleep(0.1)
                bar_counter = 0
    bar.finish()
    bar_update_counter = 0
    if norm =="fro":
        return error_fro_list_optimized_projection_DPP
    if norm =="spectral":
        return error_spectral_list_optimized_projection_DPP




def launch_exp_derandomized_volume_sampling(X_matrix,dataset_name,k,exp_number,V,D,V_k):

    #print(timeit.default_timer())
    #_,D,V = np.linalg.svd(X_matrix)
    #V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    
    #A_S = NAL.MultiRounds()
    error_fro_list_derandomized_volume_sampling= [0]*exp_number
    error_spectral_list_derandomized_volume_sampling = [0]*exp_number
    delta_time_list_derandomized_volume_sampling = [0]*exp_number
    for t in range(exp_number):
        NAL = derandomized_k_Volume_Sampling_Sampler(X_matrix, k, d)
        delta_time_list_derandomized_volume_sampling[t] = timeit.default_timer()
        print("Derandomized volume Sampling, sample number")
        print(t)
        A_S = NAL.MultiRounds()
        print(NAL.sampling_list)
        while(np.linalg.matrix_rank(A_S)<k):
            print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.MultiRounds()
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            delta_time_list_derandomized_volume_sampling[t] = timeit.default_timer() - delta_time_list_derandomized_volume_sampling[t]
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list_derandomized_volume_sampling[t]=error_temp_fro
            error_spectral_list_derandomized_volume_sampling[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
    return error_fro_list_derandomized_volume_sampling






def launch_exp_volume_sampling(X_matrix,dataset_name,k,exp_number,V,D,V_k,norm):
    print("Volume Sampling")
    #print(timeit.default_timer())
    
    #V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = k_Volume_Sampling_Sampler(X_matrix, k,D, V,d)
    A_S = NAL.MultiRounds()
    error_fro_list_volume_sampling= [0]*exp_number
    error_spectral_list_volume_sampling = [0]*exp_number
    delta_time_list_volume_sampling = [0]*exp_number
    bar = progressbar.ProgressBar(maxval=5, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bar_counter = 0
    bar_update_counter = 0
    exp_number_by_five = int(exp_number/5)
    for t in range(exp_number):
        delta_time_list_volume_sampling[t] = timeit.default_timer()
        A_S = NAL.MultiRounds()
        while(np.linalg.matrix_rank(A_S)<k):
            print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.MultiRounds()
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            delta_time_list_volume_sampling[t] = timeit.default_timer() - delta_time_list_volume_sampling[t]
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)    
            error_fro_list_volume_sampling[t]=error_temp_fro
            error_spectral_list_volume_sampling[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            if bar_counter == exp_number_by_five:
                bar_update_counter = bar_update_counter +1
                bar.update(bar_update_counter)
                sleep(0.1)
                bar_counter = 0
    bar.finish()
    bar_update_counter = 0

    if norm =="fro":
        return error_fro_list_volume_sampling
    if norm =="spectral":
        return error_spectral_list_volume_sampling

def launch_exp_uniform_sampling(X_matrix,dataset_name,k,exp_number,V,D,V_k,norm):
    print("Uniform sampling")
    #print(timeit.default_timer())
    #_,D,V = np.linalg.svd(X_matrix)
    #V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = Uniform_Sampler(X_matrix, k,d)
    A_S = NAL.MultiRounds()
    error_fro_list_uniform_sampling= [0]*exp_number
    error_spectral_list_uniform_sampling = [0]*exp_number
    delta_time_list_uniform_sampling = [0]*exp_number
    bar = progressbar.ProgressBar(maxval=5, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bar_counter = 0
    bar_update_counter = 0
    exp_number_by_five = int(exp_number/5)
    for t in range(exp_number):
        delta_time_list_uniform_sampling[t] = timeit.default_timer()
        A_S = NAL.MultiRounds()
        while(np.linalg.matrix_rank(A_S)<k):
            print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.MultiRounds()
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            delta_time_list_uniform_sampling[t] = timeit.default_timer() - delta_time_list_uniform_sampling[t]
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list_uniform_sampling[t]=error_temp_fro
            error_spectral_list_uniform_sampling[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            #print(sampling_l) 
            if bar_counter == exp_number_by_five:
                bar_update_counter = bar_update_counter +1
                bar.update(bar_update_counter)
                sleep(0.1)
                bar_counter = 0
    bar.finish()
    bar_update_counter = 0            
    #return error_fro_list_uniform_sampling
    if norm =="fro":
        return error_fro_list_uniform_sampling
    if norm =="spectral":
        return error_spectral_list_uniform_sampling


def launch_exp_largest_leveragescores_sampling(X_matrix,dataset_name,k,exp_number,V,D,V_k,norm):
    print("Largest leverage scores selection")
    #print(timeit.default_timer())
    #_,D,V = np.linalg.svd(X_matrix)
    #V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = largest_leveragescores_Sampler(X_matrix, k, V,d)
    A_S = NAL.MultiRounds()
    error_fro_list_largest_lvs_sampling= [0]*exp_number
    error_spectral_list_largest_lvs_sampling = [0]*exp_number
    delta_time_list_largest_lvs_sampling = [0]*exp_number
    bar = progressbar.ProgressBar(maxval=5, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bar_counter = 0
    bar_update_counter = 0
    exp_number_by_five = int(exp_number/5)
    for t in range(exp_number):
        delta_time_list_largest_lvs_sampling[t] = timeit.default_timer()
        A_S = NAL.MultiRounds()
        while(np.linalg.matrix_rank(A_S)<k):
            print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.MultiRounds()
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            delta_time_list_largest_lvs_sampling[t] = timeit.default_timer() - delta_time_list_largest_lvs_sampling[t]
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list_largest_lvs_sampling[t]=error_temp_fro
            error_spectral_list_largest_lvs_sampling[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            #print(sampling_l) 
            if bar_counter == exp_number_by_five:
                bar_update_counter = bar_update_counter +1
                bar.update(bar_update_counter)
                sleep(0.1)
                bar_counter = 0
    bar.finish()
    bar_update_counter = 0     
    if norm =="fro":
        return error_fro_list_largest_lvs_sampling
    if norm =="spectral":
        return error_spectral_list_largest_lvs_sampling
def launch_exp_pivoted_qr_sampling(X_matrix,dataset_name,k,exp_number,V,D,V_k,norm):
    print("Pivoted QR selection")
    #print(timeit.default_timer())
    #_,D,V = np.linalg.svd(X_matrix)
    #V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    #print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = Pivoted_QR_Sampler(X_matrix, k,d)
    A_S = NAL.MultiRounds()
    error_fro_list_pivoted_qr_sampling= [0]*exp_number
    error_spectral_list_pivoted_qr_sampling = [0]*exp_number
    delta_time_list_pivoted_qr_sampling = [0]*exp_number
    bar = progressbar.ProgressBar(maxval=5, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bar_counter = 0
    bar_update_counter = 0
    exp_number_by_five = int(exp_number/5)
    for t in range(exp_number):
        delta_time_list_pivoted_qr_sampling[t] = timeit.default_timer()
        A_S = NAL.MultiRounds()
        while(np.linalg.matrix_rank(A_S)<k):
            print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.MultiRounds()
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            delta_time_list_pivoted_qr_sampling[t] = timeit.default_timer() - delta_time_list_pivoted_qr_sampling[t]
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list_pivoted_qr_sampling[t]=error_temp_fro
            error_spectral_list_pivoted_qr_sampling[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            #print(sampling_l) 
        if bar_counter == exp_number_by_five:
            bar_update_counter = bar_update_counter +1
            bar.update(bar_update_counter)
            sleep(0.1)
            bar_counter = 0
    bar.finish()
    bar_update_counter = 0               
    if norm =="fro":
        return error_fro_list_pivoted_qr_sampling
    if norm =="spectral":
        return error_spectral_list_pivoted_qr_sampling