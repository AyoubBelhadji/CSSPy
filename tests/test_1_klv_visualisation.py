import sys
sys.path.insert(0, '..')
from CSSPy.dataset_tools import *
from CSSPy.visualization_tools import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
# This is a test for basic functions of this package:
## * Calculating the k-leverage scores 
## * Calculating the p_eff(theta) function
## * Plots of k-leverage scores and cumulative k-leverage scores

# Importing the dataset
dataset_name = "Basehock"
dataset_file = dataset_name+str("_X")
t = timeit.Timer('char in text', setup='text = "sample string"; char = "g"')
X_df = pd.read_csv('datasets/'+dataset_file+'.csv', sep=",", header=None)
X_matrix = X_df.values


# Store the dimensions of the matrix
d = np.shape(X_matrix)[1]
N = np.shape(X_matrix)[0]

# SVD of the matrix
_,D,V = np.linalg.svd(X_matrix)

# The order of PCA
k = 10

# Calculate and sort the k-leverage scores
V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
klv_test_1 = np.asarray(list(reversed(np.sort((np.diag(np.dot(np.transpose(V_k),V_k)))))))

# Plot of the k-leverage scores and the cumulative k-leverage scores
#plot_leverage_scores(klv_test_1,dataset_name,k)
plot_cumul_leverage_scores(klv_test_1,dataset_name,k)