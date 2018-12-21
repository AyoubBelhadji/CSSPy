import sys
sys.path.insert(0, '..')
from CSSPy.dataset_tools import *
from CSSPy.visualization_tools import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# This is a test for basic functions of this package:
## * Calculating the k-leverage scores 
## * Calculating the p_eff(theta) function
## * Plots of k-leverage scores and cumulative k-leverage scores

# Importing the dataset
import scipy.io
import numpy as np

data = scipy.io.loadmat("datasets/gisette.mat")

for i in data:
	if '__' not in i and 'readme' not in i:
		np.savetxt(("datasets/gisette.csv"),data[i],delimiter=',')