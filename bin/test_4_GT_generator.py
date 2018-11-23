import sys
#sys.path.append('../')
sys.path.insert(0, '..')
from CSSPy.dataset_tools import *
#from CSSPy.visualization_tools import *
from CSSPy.derandomized_projection_dpp_sampler import *
from CSSPy.derandomized_volume_sampler import *
from CSSPy.volume_sampler import *
from CSSPy.optimized_projection_dpp_sampler import *
from CSSPy.projection_dpp_sampler import *
from CSSPy.uniform_sampler import *
from CSSPy.evaluation_functions import *
from CSSPy.experiments_tools import *
from CSSPy.k_means import *
from CSSPy.visualization_tools import *

import numpy as np
#import matplotlib.pyplot as plt
import timeit
import pandas as pd
from matplotlib import pyplot as plt
# This is a test for subsampling functions:
## * Projection DPPs
## * Volume sampling
## * Uniform sampling

# Import two datasets
dataset_name = "colon"
dataset_file = dataset_name+str("_X")
clustering_labels_file = dataset_name +str("_Y")
t = timeit.Timer('char in text', setup='text = "sample string"; char = "g"')
X_df = pd.read_csv('datasets/'+dataset_file+'.csv', sep=",", header=None)

X_matrix = X_df.values

d = np.shape(X_matrix)[1]
N = np.shape(X_matrix)[0] -1

_,D,V = np.linalg.svd(X_matrix)

k = 10


V_k_ = calculate_right_eigenvectors_k_svd(X_matrix,k)

V_k = V_k_ #V[0:k,:]
#np.asarray(list(reversed(np.sort(lv_scores_vector))))


rank_X = np.shape(D)[0]


klv_test_1 = np.asarray(list(reversed(np.sort((np.diag(np.dot(np.transpose(V_k),V_k)))))))


plot_leverage_scores(klv_test_1,dataset_name,k)

plot_cumul_leverage_scores(klv_test_1,dataset_name,k)
