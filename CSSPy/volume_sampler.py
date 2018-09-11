
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


class k_Volume_Sampling_Sampler:
    def __init__(self, A, k,D, Q,N):
        self.A = A
        self.D = D
        self.nnz_D = np.count_nonzero(D)
        self.Q = np.transpose(Q[0:self.nnz_D,:])
        self.V = Q
        self.N = N
        self.k = k
        self.sampling_list = []
        self.column_selected_temp = np.zeros(k)
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
        sampled_indices_ = k_dpp_KuTa12(np.power(self.D,2), np.transpose(self.V[0:self.nnz_D,:]), self.k)
        return self.A[:,sampled_indices_]


def k_dpp_KuTa12(eig_val, eig_vec, k):
    """ Sample from :math:`\operatorname{k-DPP}(K)` using the eigen-decomposition of the
    similarity kernel :math:`K`.
    :param eig_val: 
        Collection of eigen values of the similarity kernel :math:`K`.
    :type eig_val: 
        list
    :param eig_vec: 
        Eigen-vectors of the similarity kernel :math:`K`.
    :type eig_vec: 
        array_type
    
    :param k: 
        Size of the sample.
    :type k: 
        int
    :return: 
        A sample from :math:`\operatorname{k-DPP}(K)`.
    :rtype: 
        list
            
    .. seealso::
        Algorithm 1, 8 in :cite:`KuTa12` 
        - :func:`dpp_exact_sampling_KuTa12 <dpp_exact_sampling_KuTa12>`
        - :func:`k_dpp_TrBaAm17_Wood <k_dpp_TrBaAm17_Wood>`
        - :func:`k_dpp_TrBaAm17_Fs <k_dpp_TrBaAm17_Fs>`
    """

    E = elem_symm_poly(eig_val, k) # Evaluate the elem symm polys in the eig_values 
    N = len(eig_val) # Size of the ground set
    S = [] # Initialization of the output
    l = k # Size of the sample

    for n in range(N,0,-1):
        if l == 0:
            break
        if np.random.rand() < eig_val[n-1]*(E[l-1,n-1]/E[l,n]):
            S.append(n-1)
            l-=1

    eig_v = np.zeros(N)
    eig_v[S] = 1.0

    return dpp_exact_sampling_KuTa12(eig_v, eig_vec)



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


def dpp_exact_sampling_KuTa12(eig_val, eig_vec):
    """ Sample from :math:`\operatorname{DPP}(K)` using the eigen-decomposition of the
    similarity kernel :math:`K`. 
    It is based on the orthogonalization of the selected eigen-vectors.
    :param eig_val: 
        Collection of eigen values of the similarity kernel :math:`K`.
    :type eig_val: 
        list
    :param eig_vec: 
        Eigen-vectors of the similarity kernel :math:`K`.
    :type eig_vec: 
        array_type
        
    :return: 
        A sample from :math:`\operatorname{DPP}(K)`.
    :rtype: 
        list
    .. seealso::
        Algorithm 1 in :cite:`KuTa12`
        - :func:`dpp_exact_sampling_Wood <dpp_exact_sampling_Wood>`
        - :func:`dpp_exact_sampling_GS <dpp_exact_sampling_GS>`
    """

    # Phase 1: 
    # Select eigen vectors \propto eig_vals
    ind_bool = np.random.rand(len(eig_val)) < eig_val
    # Stack the selected eigen-vectors
    V = eig_vec[:,ind_bool]
    # N = size of the ground set, n = size of the sample
    N, n = V.shape 

    # Phase 2: Chain rule, to be clarified
    # Initialize the sample
    Y = []
    
    # Following [Algo 1, KuTa12], the aim is to compute the orhto complement of the subspace spanned by the selected
    # eigen-vectors to the canonical vectors \{e_i ; i \in Y\}. We proceed recursively.
    for it in range(1,n+1):
        
        norms_2 = np.sum(np.square(V), axis=1) 
        # Pick an item
        i = np.random.choice(N, 1, p=norms_2/(n-it+1))[0] 
        # Add the item just picked
        Y.append(i) 

        # Cancel the contribution of e_i to the remaining vectors that is, 
        # find the subspace of V that is orthogonal to \{e_i ; i \in Y\}
        if it<n:
            # Take the index of a vector that has a non null contribution along e_i
            j = np.where(V[i,:]!=0)[0][0]
            # Cancel the contribution of the remaining vectors along e_i, but stay in the subspace spanned by V
            # i.e. get the subspace of V orthogonal to \{e_i ; i \in Y\}
            V -= (V[:,j]/V[i,j])[:,np.newaxis] * V[i,:] 
            # V_j is set to 0 so we delete it and we can
            # derive an orthononormal basis of the subspace under consideration
            V, R = np.linalg.qr(np.delete(V, j, axis=1)) 
            #print(np.shape(V))
    
    return sorted(Y)
