import sys
#sys.path.append('../')
sys.path.insert(0, '..')
from CSSPy.dataset_tools import *
#from CSSPy.visualization_tools import *

from CSSPy.evaluation_functions import *
from CSSPy.experiments_tools import *
from CSSPy.k_means import *
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
dataset_name = "arcene"
dataset_file = "arcene_train.data"
#clustering_labels_file = "colon"+str("_Y")

#COLUMNS_COUNT = 89
# read first `COLUMNS_COUNT` lines to serve as a header
#with open('datasets/arcene_train.data', 'r') as f:
#    columns = [next(f).strip() for line in range(COLUMNS_COUNT)]
# read rest of the file to temporary DataFrame
temp_df = pd.read_csv('datasets/arcene_train.data', header=None, delimiter=' ', skip_blank_lines=True)
# split temp DataFrame on even and odd rows
#even_df = temp_df.iloc[::2].reset_index(drop=True)
#odd_df = temp_df.iloc[1::2].reset_index(drop=True)
# stack even and odd DataFrames horizontaly
#df = pd.concat([even_df, odd_df], axis=1)
# assign column names
#temp_df.columns = columns
# save result DataFrame to csv
temp_df.to_csv('out.csv', index=False)






#t = timeit.Timer('char in text', setup='text = "sample string"; char = "g"')
#X_df = pd.read_csv('datasets/'+dataset_file, sep=",", header=None)
##Y_df = pd.read_csv('datasets/'+clustering_labels_file+'.csv', sep=",", header=None)



X_matrix = temp_df.values
#Y_matrix = (-Y_df.values +1)/2
#Y_matrix_2 = (Y_df.values +1)/2
##mean_X = X_matrix.mean(axis=0)
##X_matrix = X_matrix - mean_X
#print(np.shape(X_matrix))
#d = np.shape(X_matrix)[1]
#N = np.shape(X_matrix)[0] -1
#t_1 = timeit.default_timer() 

##_,D,V = np.linalg.svd(X_matrix)

#U,D,V = svds(X_matrix, 61 , return_singular_vectors='true')

