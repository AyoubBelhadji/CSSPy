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

def expected_approximation_error_fro_for_sampling_scheme(X,U,k,N):
    ## X is the matrix X :)
    ## U is the matrix used in the sampling: we sample propotional to the volume of UU^{T}_{S,S}: 
    ## we are not sampling but we need the weigth to estimate the expected error
    ## k is the rank of the approximation
    ## N is the number of columns (to be changed to avoid confusion with the number of points)
    _,Sigma,_ = np.linalg.svd(X, full_matrices=False)
    ## Estimating the spectrum of X -> needed in approximation_error_function_fro
    volumes_array = [np.abs(np.linalg.det(np.dot(U[:,list(comb)].T,U[:,list(comb)]))) for comb in combinations(range(N),k)]
    ## Construct the array of weights: the volumes of UU^{T}_{S,S}
    volumes_array_sum = np.sum(volumes_array)
    ## The normalization constant
    volumes_array = volumes_array/volumes_array_sum
    ## The weigths normalized
    approximation_error_array = [approximation_error_function_fro(Sigma,k,X,X[:,list(comb)]) for comb in combinations(range(N),k)]
    ## Calculating the approximation error for every k-tuple
    expected_value = np.dot(approximation_error_array,volumes_array)
    ## The expected value of the approximatione error is just the dot product of the two arrays above
    return expected_value


def approximation_error_function_fro(Sigma,k,X,X_S):
    ## Sigma is the spectrum of the matrix X: we need to calculate the optimal approximation error given by the PCA
    ## k is the rank of the approximation
    ## X is the initial matrix
    ## X_S is the subset of columns of the matrix X for witch we calculate the approximation error ratio
    d = list(Sigma.shape)[0] # the dimension of the matrix X
    Sigma = np.multiply(Sigma,Sigma)  # Sigma power 2 -> we are intersted in the approximation error square
    sigma_S_temp = np.linalg.inv(np.dot(X_S.T,X_S))  # just a temporary matrix to construct the projection matrix
    projection_S = np.dot(np.dot(X_S,sigma_S_temp),X_S.T) # the projection matrix P_S
    res_X = X - np.dot(projection_S,X) # The projection of the matrix X in the orthogonal of S
    approximation_error_ratio = np.power(np.linalg.norm(res_X,'fro'),2)/np.sum(Sigma[k:d])
    # Calculate the apparoximation error ratio
    return approximation_error_ratio


def approximation_error_function_spectral(Sigma,k,X,X_S):
    ## Sigma is the spectrum of the matrix X: we need to calculate the optimal approximation error given by the PCA
    ## k is the rank of the approximation
    ## X is the initial matrix
    ## X_S is the subset of columns of the matrix X for witch we calculate the approximation error ratio
    d = list(Sigma.shape)[0] # the dimension of the matrix X
    Sigma = np.multiply(Sigma,Sigma)  # Sigma power 2 -> we are intersted in the approximation error square
    sigma_S_temp = np.linalg.inv(np.dot(X_S.T,X_S))  # just a temporary matrix to construct the projection matrix
    projection_S = np.dot(np.dot(X_S,sigma_S_temp),X_S.T) # the projection matrix P_S
    res_X = X - np.dot(projection_S,X) # The projection of the matrix X in the orthogonal of S
    approximation_error_ratio = np.power(np.linalg.norm(res_X,ord = 2),2)/np.sum(Sigma[k:k+1])
    # Calculate the apparoximation error ratio
    return approximation_error_ratio

def upper_bound_error_function_for_projection_DPP(k,X,X_S):
    ## Sigma is the spectrum of the matrix X: we need to calculate the optimal approximation error given by the PCA
    ## k is the rank of the approximation
    ## X is the initial matrix
    ## X_S is the subset of columns of the matrix X for witch we calculate the approximation error ratio
    _,sigma_S_temp,_ = np.linalg.svd(X_S, full_matrices=False)  # just a temporary matrix to construct the projection matrix
    trunc_product = np.power(np.prod(sigma_S_temp[0:k-1]),2)
    if np.power(np.prod(sigma_S_temp[0:k]),2) == 0:
        trunc_product = 0
    # Calculate the apparoximation error ratio
    return trunc_product

def tight_upper_bound_error_function_fro(k,X,X_S,V_k,V_k_S):
    ## Sigma is the spectrum of the matrix X: we need to calculate the optimal approximation error given by the PCA
    ## k is the rank of the approximation
    ## X is the initial matrix
    ## X_S is the subset of columns of the matrix X for witch we calculate the approximation error ratio
    _,Sigma,_ = np.linalg.svd(X, full_matrices=False)
    d = list(Sigma.shape)[0]
    Sigma = np.multiply(Sigma,Sigma)
    if np.linalg.matrix_rank(V_k_S,0.0001) == k:
        temp_T = np.dot(np.linalg.inv(V_k_S),V_k)
        temp_matrix = X - np.dot(X_S,temp_T)
    
        return np.power(np.linalg.norm(temp_matrix,'fro'),2)/np.sum(Sigma[k:d])
    else:
        return 0
    


def get_p_eff_leverage_scores(Q,k):
    lv_scores_vector = estimate_leverage_scores_from_orthogonal_matrix(Q)
    p = 0
    lv_scores_sum_until_p = 0
    while lv_scores_sum_until_p < k-1/2:
        lv_scores_sum_until_p = lv_scores_sum_until_p+lv_scores_vector[p]
        p = p +1
    return p


def estimate_leverage_scores_from_orthogonal_matrix(Q):
    [N,_] = np.shape(Q)
    lv_scores_vector = np.zeros((N,1))
    lv_scores_vector = np.diag(np.dot(Q,np.transpose(Q)))
    lv_scores_vector = np.asarray(list(reversed(np.sort(lv_scores_vector))))
    return lv_scores_vector


def from_p_eff_to_error_bound(list_of_p,k,d):
    list_res = []
    for p in list_of_p:
        new_element = 1+(p-k)/(d-k)*k
        list_res.append(new_element)
    return list_res


def from_p_eff_to_error_bound_2(list_of_p,k,d):
    list_res = []
    for p in list_of_p:
        new_element = 1+(p-k+1)/(d-k)*(k+1)
        list_res.append(new_element)
    return list_res
