import sys
#sys.path.append('../')
sys.path.insert(0, '..')
from CSSPy.dataset_tools import *
from CSSPy.doublephase_sampler import *
from CSSPy.derandomized_projection_dpp_sampler import *
from CSSPy.derandomized_volume_sampler import *
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
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import random, linalg, dot, diag, all, allclose
import timeit
from scipy.sparse.linalg import svds


def launch_exp_derandomization_projection_dpp(X_matrix,dataset_name,k,exp_number):

    print(timeit.default_timer())
    _,D,V = np.linalg.svd(X_matrix)
    V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    print(timeit.default_timer())
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


def launch_exp_double_phase_sampler(X_matrix,dataset_name,k,exp_number):

    print(timeit.default_timer())
    _,D,V = np.linalg.svd(X_matrix)
    V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = double_Phase_Sampler(X_matrix, k, V_k,d,20*k)
    A_S = NAL.DoublePhase()
    print("A_S")
    #print(A_S)
    error_fro_list = [0]*exp_number
    error_spectral_list = [0]*exp_number
    delta_time_list = [0]*exp_number
    for t in range(exp_number):
        NAL = double_Phase_Sampler(X_matrix, k, V_k,d,20*k)
        delta_time_list[t] = timeit.default_timer()
        print("Double Phase, sample number")
        print(t)
        A_S = NAL.DoublePhase()
        print(NAL.sampling_list)
        while(np.linalg.matrix_rank(A_S)<k):
            print("fail")
            #NAL_1 = projection_DPP_Sampler(X_matrix, k, V_k,d)
            A_S = NAL.DoublePhase()
            #sampling_l = NAL_1.sampling_list
        if np.linalg.matrix_rank(A_S)==k:
            delta_time_list[t] = timeit.default_timer() - delta_time_list[t] 
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list[t]=error_temp_fro
            error_spectral_list[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
    return error_fro_list




def launch_exp_projection_dpp(X_matrix,dataset_name,k,exp_number):

    print(timeit.default_timer())
    _,D,V = np.linalg.svd(X_matrix)
    V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = projection_DPP_Sampler(X_matrix, k, V_k,d)
    A_S = NAL.MultiRounds()
    error_fro_list_projection_DPP = [0]*exp_number
    error_spectral_list_projection_DPP = [0]*exp_number
    delta_time_list_projection_DPP = [0]*exp_number
    for t in range(exp_number):
        delta_time_list_projection_DPP[t] = timeit.default_timer()
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
            delta_time_list_projection_DPP[t] = timeit.default_timer() - delta_time_list_projection_DPP[t] 
            error_temp_fro = approximation_error_function_fro(k,X_matrix,A_S,D)
            error_temp_spectral = approximation_error_function_spectral(k,X_matrix,A_S,D)        
            error_fro_list_projection_DPP[t]=error_temp_fro
            error_spectral_list_projection_DPP[t]=error_temp_spectral
            sampling_l = NAL.sampling_list
            #print(error_temp_fro)
    return error_fro_list_projection_DPP



def launch_exp_optimized_projection_dpp(X_matrix,dataset_name,k,exp_number):

    print(timeit.default_timer())
    _,D,V = np.linalg.svd(X_matrix)
    V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = optimized_projection_DPP_Sampler(X_matrix, k,D, V,d)
    A_S = NAL.MultiRounds()
    error_fro_list_optimized_projection_DPP = [0]*exp_number
    error_spectral_list_optimized_projection_DPP = [0]*exp_number
    delta_time_list_optimized_projection_DPP = [0]*exp_number
    for t in range(exp_number):
        delta_time_list_optimized_projection_DPP[t] = timeit.default_timer()
        print("Optimized Projection DPP, sample number")
        print(t)
        A_S = NAL.MultiRounds()
        print(NAL.sampling_list)
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
    return error_fro_list_optimized_projection_DPP
def launch_exp_derandomized_volume_sampling(X_matrix,dataset_name,k,exp_number):

    print(timeit.default_timer())
    _,D,V = np.linalg.svd(X_matrix)
    V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    print(timeit.default_timer())
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






def launch_exp_volume_sampling(X_matrix,Y_matrix,dataset_name,k,exp_number):

    print(timeit.default_timer())
    _,D,V = np.linalg.svd(X_matrix)
    V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = k_Volume_Sampling_Sampler(X_matrix, k,D, V,d)
    A_S = NAL.MultiRounds()
    error_fro_list_volume_sampling= [0]*exp_number
    error_spectral_list_volume_sampling = [0]*exp_number
    delta_time_list_volume_sampling = [0]*exp_number
    for t in range(exp_number):
        delta_time_list_volume_sampling[t] = timeit.default_timer()
        print("Volume Sampling, sample number")
        print(t)
        A_S = NAL.MultiRounds()
        print(NAL.sampling_list)
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
    return error_fro_list_volume_sampling


def launch_exp_uniform_sampling(X_matrix,dataset_name,k,exp_number):

    print(timeit.default_timer())
    _,D,V = np.linalg.svd(X_matrix)
    V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = Uniform_Sampler(X_matrix, k,d)
    A_S = NAL.MultiRounds()
    error_fro_list_uniform_sampling= [0]*exp_number
    error_spectral_list_uniform_sampling = [0]*exp_number
    delta_time_list_uniform_sampling = [0]*exp_number
    for t in range(exp_number):
        delta_time_list_uniform_sampling[t] = timeit.default_timer()
        print("Uniform sampling selection, sample number")
        print(t)
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
            
    return error_fro_list_uniform_sampling

def launch_exp_largest_leveragescores_sampling(X_matrix,dataset_name,k,exp_number):

    print(timeit.default_timer())
    _,D,V = np.linalg.svd(X_matrix)
    V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = largest_leveragescores_Sampler(X_matrix, k, V,d)
    A_S = NAL.MultiRounds()
    error_fro_list_largest_lvs_sampling= [0]*exp_number
    error_spectral_list_largest_lvs_sampling = [0]*exp_number
    delta_time_list_largest_lvs_sampling = [0]*exp_number
    for t in range(exp_number):
        delta_time_list_largest_lvs_sampling[t] = timeit.default_timer()
        print("Largest leverage scores selection, sample number")
        print(t)
        A_S = NAL.MultiRounds()
        print(NAL.sampling_list)
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
            
    return error_fro_list_largest_lvs_sampling

def launch_exp_pivoted_qr_sampling(X_matrix,dataset_name,k,exp_number):

    print(timeit.default_timer())
    _,D,V = np.linalg.svd(X_matrix)
    V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
    print(timeit.default_timer())
    d = np.shape(X_matrix)[1]
        
    NAL = Pivoted_QR_Sampler(X_matrix, k,d)
    A_S = NAL.MultiRounds()
    error_fro_list_pivoted_qr_sampling= [0]*exp_number
    error_spectral_list_pivoted_qr_sampling = [0]*exp_number
    delta_time_list_pivoted_qr_sampling = [0]*exp_number
    for t in range(exp_number):
        delta_time_list_pivoted_qr_sampling[t] = timeit.default_timer()
        print("Largest leverage scores selection, sample number")
        print(t)
        A_S = NAL.MultiRounds()
        print(NAL.sampling_list)
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
            
    return error_fro_list_pivoted_qr_sampling