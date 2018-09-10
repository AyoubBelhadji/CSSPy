import sys
#sys.path.append('../')
sys.path.insert(0, '..')
from CSSPy.dataset_tools import *
from CSSPy.visualization_tools import *
import numpy as np
#import matplotlib.pyplot as plt
import timeit
import pandas as pd
from matplotlib import pyplot as plt
# This is a test for basic functions of this package:
## * Calculating the k-leverage scores 
## * Calculating the p_eff(theta) function
## * Plots of k-leverage scores and cumulative k-leverage scores

# Import two datasets
dataset_name = "BASEHOCK"
dataset_file = "BASEHOCK"+str("_X")
t = timeit.Timer('char in text', setup='text = "sample string"; char = "g"')
X_df = pd.read_csv('datasets/'+dataset_file+'.csv', sep=",", header=None)
X_matrix = X_df.values
mean_X = X_matrix.mean(axis=0)
X_matrix = X_matrix - mean_X
print(np.shape(X_matrix))
d = np.shape(X_matrix)[1]
N = np.shape(X_matrix)[0] -1
t_1 = timeit.default_timer() 

_,D,V = np.linalg.svd(X_matrix)
print(timeit.default_timer())

V_k_ = calculate_right_eigenvectors_k_svd(X_matrix,k)
print(timeit.default_timer())
V_k = V_k_ #V[0:k,:]
#np.asarray(list(reversed(np.sort(lv_scores_vector))))

k = 5
rank_X = np.shape(D)[0]
print("classic bound for spectral norm")
print(np.sqrt((d-k)*k))
print("new bound for spectral norm")
#from  sys  import  path

#for d in sys.path:
#    print(d)



klv_test_1 = np.asarray(list(reversed(np.sort((np.diag(np.dot(np.transpose(V_k),V_k)))))))

plot_singular_values("matrix",X_matrix,dataset_name,k)

plot_leverage_scores("vector",klv_test_1,dataset_name,k)
plot_cumulative_leverage_scores("vector",klv_test_1,dataset_name,k)