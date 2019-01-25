# This is a comparison for the CSSP algorithms on real datasets.
# This is a test for subsampling functions:
## * Projection DPPs
## * Volume sampling
## * Pivoted QR
## * Double Phase
## * Largest leverage scores

##
import sys

sys.path.insert(0, '..')
from CSSPy.dataset_tools import *
from CSSPy.volume_sampler import *
from CSSPy.optimized_projection_dpp_sampler import *
from CSSPy.projection_dpp_sampler import *
from CSSPy.uniform_sampler import *
from CSSPy.evaluation_functions import *
from CSSPy.experiments_tools import *
from CSSPy.visualization_tools import *

import numpy as np
import timeit
import pandas as pd
from matplotlib import pyplot as plt


# Import the dataset
dataset_name = "colon"
dataset_file = dataset_name+str("_X")
clustering_labels_file = dataset_name +str("_Y")
t = timeit.Timer('char in text', setup='text = "sample string"; char = "g"')
X_df = pd.read_csv('datasets/'+dataset_file+'.csv', sep=",", header=None)
X_matrix = X_df.values

# The dimensions of the matrix
d = np.shape(X_matrix)[1]
N = np.shape(X_matrix)[0] -1

# The singular value decomposition of the matrix
k = 10
_,D,V = np.linalg.svd(X_matrix)
V_k = calculate_right_eigenvectors_k_svd(X_matrix,k)
rank_X = np.shape(D)[0]



# Calculate and sort the k-leverage scores
klv_test_1 = np.asarray(list(reversed(np.sort((np.diag(np.dot(np.transpose(V_k),V_k)))))))
# Plot of the k-leverage scores and the cumulative k-leverage scores
plot_cumul_leverage_scores(klv_test_1,dataset_name,k)



def random_error_list_to_min_error_list(error_list):
    l_test = len(error_list)
    min_value_random_list = min(error_list)
    new_list = [min_value_random_list]*l_test
    return min_value_random_list



# These lists contains the approximation errors after boosting
boosting_error_fro_volume_sampling_list = []
boosting_error_fro_projection_dpp_list = []
boosting_error_fro_largest_lvs_list = []
boosting_error_fro_pivoted_qr_list = []
boosting_error_fro_double_phase_list = []

boosting_error_fro_aggregated_list = []
# This is the aggregated list for the results of all the algortihms
error_fro_aggregated_list = []

# This parameter equals 0 for the first iteration and 1 for the rest to avoid the repetition of the determinstic sampling
deterministic_algos_flag = 0

# Initialization of the deterministic algorithms
error_fro_list_pivoted_qr_sampling = 0
error_fro_list_largest_lvs_sampling = 0

# Launch the simulations with the following batch sizes
exp_number = 50
boosting_batch = 1

for T_B in list(range(boosting_batch)):
    print ("Boosting step")
    print(T_B)
    error_fro_aggregated_list = []
    error_fro_list_double_phase_sampling = launch_exp_double_phase_sampler(X_matrix,dataset_name,k,exp_number,V,D,V_k,"fro") 
    error_fro_list_optimized_projection_DPP = launch_exp_optimized_projection_dpp(X_matrix,dataset_name,k,exp_number,V,D,V_k,"fro")    
    error_fro_list_volume_sampling = launch_exp_volume_sampling(X_matrix,dataset_name,k,exp_number,V,D,V_k,"fro")    

    if deterministic_algos_flag == 0:
        error_fro_list_pivoted_qr_sampling = launch_exp_pivoted_qr_sampling(X_matrix,dataset_name,k,exp_number,V,D,V_k,"fro")
        error_fro_list_largest_lvs_sampling = launch_exp_largest_leveragescores_sampling(X_matrix,dataset_name,k,exp_number,V,D,V_k,"fro")
        min_error_fro_list_pivoted_qr_sampling = error_fro_list_pivoted_qr_sampling[0]
        min_error_fro_list_largest_lvs_sampling = error_fro_list_largest_lvs_sampling[0]
    
    deterministic_algos_flag = deterministic_algos_flag +1   
    

    min_error_fro_list_volume_sampling = random_error_list_to_min_error_list(error_fro_list_volume_sampling)
    min_error_fro_list_optimized_projection_DPP = random_error_list_to_min_error_list(error_fro_list_optimized_projection_DPP)
    min_error_fro_list_double_phase_sampling = random_error_list_to_min_error_list(error_fro_list_double_phase_sampling)
    
    min_error_fro_list_largest_lvs_sampling = error_fro_list_largest_lvs_sampling[0]
    min_error_fro_list_pivoted_qr_sampling = error_fro_list_pivoted_qr_sampling[0]

    boosting_error_fro_volume_sampling_list.append(min_error_fro_list_volume_sampling)
    boosting_error_fro_projection_dpp_list.append(min_error_fro_list_optimized_projection_DPP)
    boosting_error_fro_largest_lvs_list.append(min_error_fro_list_largest_lvs_sampling)
    boosting_error_fro_pivoted_qr_list.append(min_error_fro_list_pivoted_qr_sampling)
    boosting_error_fro_double_phase_list.append(min_error_fro_list_double_phase_sampling)
    

    error_fro_aggregated_list.append(error_fro_list_volume_sampling)
    error_fro_aggregated_list.append(error_fro_list_optimized_projection_DPP)
    error_fro_aggregated_list.append(error_fro_list_largest_lvs_sampling)
    error_fro_aggregated_list.append(error_fro_list_pivoted_qr_sampling)
    error_fro_aggregated_list.append(error_fro_list_double_phase_sampling)


boosting_error_fro_aggregated_list.append(boosting_error_fro_volume_sampling_list)
boosting_error_fro_aggregated_list.append(boosting_error_fro_projection_dpp_list)
boosting_error_fro_aggregated_list.append(boosting_error_fro_largest_lvs_list)
boosting_error_fro_aggregated_list.append(boosting_error_fro_pivoted_qr_list)
boosting_error_fro_aggregated_list.append(boosting_error_fro_double_phase_list)

# Plot the comparison of the algorithms 

plt.figure(figsize=(10, 6)) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax = plt.subplot(111) 
box1 =plt.boxplot(error_fro_aggregated_list, showfliers=False)
plt.setp(box1['medians'], color='red', linewidth=3)

plt.ylabel(r'$\mathrm{\|\| X- \pi_{C} X \|\| _{Fr}}$', fontsize=16)
plt.gca().xaxis.set_ticklabels(["Volume S.","Projection DPP","Largest lvs","Pivoted QR","Double Phase"])

# Save the results on a txt file
savefile_name = "results/test_2/"+dataset_name+"_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".txt"
np.savetxt(savefile_name, error_fro_aggregated_list, fmt='%f')

# Save the figure on a pdf file
figfile_name= "results/test_2/"+dataset_name+"_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".pdf"
plt.savefig(figfile_name)

# Show the figure
plt.show()


# Plot the comparison of boosting of the algorithms 


plt.figure(figsize=(10, 6)) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax = plt.subplot(111) 
box_2 = plt.boxplot(boosting_error_fro_aggregated_list, showfliers=False)
plt.setp(box_2['medians'], color='red', linewidth=3)

plt.ylabel(r'$\mathrm{\|\| X- \pi_{C} X \|\| _{Fr}}$', fontsize=16)
 

plt.gca().xaxis.set_ticklabels(["Volume S.","Projection DPP","Largest lvs","Pivoted QR","Double Phase"])

# Save the results on a txt file
savefile_name = "results/test_2/"+dataset_name+"_boosting_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".txt"
np.savetxt(savefile_name, boosting_error_fro_aggregated_list, fmt='%f')

# Save the figure on a pdf file
figfile_name= "results/test_2/"+dataset_name+"_boosting_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".pdf"
plt.savefig(figfile_name)

# Show the figure
plt.show()




