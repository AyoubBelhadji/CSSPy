# This is a visualization script for the CSSP results on real datasets.
# The visualization is available for the following subsampling functions:
## * Projection DPPs
## * Volume sampling
## * Pivoted QR
## * Double Phase
## * Largest leverage scores

import sys
sys.path.insert(0, '..')
from CSSPy.dataset_tools import *
from CSSPy.visualization_tools import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Importing the dataset
dataset_name = "leukemia"

exp_number = 50
k = 10

# Load the results from a txt file

savefile_name = "results/test_2/boxplots/"+dataset_name+"_boosting_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".txt"
boosting_error_fro_aggregated_list_load = np.loadtxt(savefile_name)

boosting_error_fro_aggregated_list_load_to_list = []
for i in list(range(5)):
    boosting_error_fro_aggregated_list_load_to_list.append(boosting_error_fro_aggregated_list_load[i])

# Plot the comparison of boosting of the algorithms 

plt.figure(figsize=(10, 8)) 
plt.xticks(fontsize=22)
plt.yticks(fontsize=16)
ax = plt.subplot(111) 
box_2 = plt.boxplot(boosting_error_fro_aggregated_list_load_to_list, showfliers=False)
plt.setp(box_2['medians'], color='red', linewidth=3)
plt.ylabel(r'$\mathrm{\|\| X- \pi_{S}^{Fr} X \|\| _{Fr}}$', fontsize=18)
plt.xticks(rotation=45)
plt.gca().xaxis.set_ticklabels(["Volume S.","Projection DPP","Largest lvs","Pivoted QR","Double Phase"])

# Save the figure on a pdf file

figfile_name= "results/test_2/"+dataset_name+"_boosting_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".pdf"
plt.savefig(figfile_name)
plt.show()

### The boosting of the algorithms
# Load the results from a txt file

savefile_name = "results/test_2/boxplots/"+dataset_name+"_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".txt"
error_fro_aggregated_list_load = np.loadtxt(savefile_name)
error_fro_aggregated_list_load_to_list = []
for i in list(range(5)):
    error_fro_aggregated_list_load_to_list.append(error_fro_aggregated_list_load[i])

# Plot the comparison of boosting of the algorithms 

plt.figure(figsize=(10, 8)) 
plt.xticks(fontsize=22)
plt.yticks(fontsize=16)
ax = plt.subplot(111) 
box_2 = plt.boxplot(error_fro_aggregated_list_load_to_list, showfliers=False)
plt.setp(box_2['medians'], color='red', linewidth=3)
plt.ylabel(r'$\mathrm{\|\| X- \pi_{S}^{Fr} X \|\| _{Fr}}$', fontsize=18)
plt.xticks(rotation=45)
plt.gca().xaxis.set_ticklabels(["Volume S.","Projection DPP","Largest lvs","Pivoted QR","Double Phase"])

# Save the figure on a pdf file

figfile_name= "results/test_2/"+dataset_name+"_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".pdf"
plt.savefig(figfile_name)

plt.show()