import sys
#sys.path.append('../')
sys.path.insert(0, '..')
from CSSPy.dataset_tools import *
from CSSPy.visualization_tools import *
from CSSPy.derandomized_projection_dpp_sampler import *
from CSSPy.derandomized_volume_sampler import *
from CSSPy.volume_sampler import *
from CSSPy.optimized_projection_dpp_sampler import *
from CSSPy.projection_dpp_sampler import *
from CSSPy.uniform_sampler import *
from CSSPy.evaluation_functions import *
from CSSPy.experiments_tools import *
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

#_,D,V = np.linalg.svd(X_matrix)

_,D,V = svds(X_matrix, 400 , return_singular_vectors='true')

k = 10
t_2 = timeit.default_timer()
print(t_2-t_1)

V_k_ = calculate_right_eigenvectors_k_svd(X_matrix,k)
t_3 = timeit.default_timer()
print(t_3-t_2)
V_k = V_k_ #V[0:k,:]
#np.asarray(list(reversed(np.sort(lv_scores_vector))))


rank_X = np.shape(D)[0]
#print("classic bound for spectral norm")
#print(np.sqrt((d-k)*k))
#print("new bound for spectral norm")
#from  sys  import  path

#for d in sys.path:
#    print(d)

exp_number = 100

klv_test_1 = np.asarray(list(reversed(np.sort((np.diag(np.dot(np.transpose(V_k),V_k)))))))

##plot_singular_values("matrix",X_matrix,dataset_name,k)
##plot_leverage_scores("vector",klv_test_1,dataset_name,k)
##plot_cumulative_leverage_scores("vector",klv_test_1,dataset_name,k)
##error_fro_list_derandomized_projection_dpp_sampling = launch_exp_derandomization_projection_dpp(X_matrix,dataset_name,k,exp_number) 
#error_fro_list_double_phase_sampling = launch_exp_double_phase_sampler(X_matrix,dataset_name,k,exp_number) 
##error_fro_list_derandomized_volume_sampling = launch_exp_derandomized_volume_sampling(X_matrix,dataset_name,k,exp_number)    
#error_fro_list_projection_DPP = launch_exp_projection_dpp(X_matrix,dataset_name,k,exp_number)    
#error_fro_list_pivoted_qr_sampling = launch_exp_pivoted_qr_sampling(X_matrix,dataset_name,k,exp_number)
#error_fro_list_largest_lvs_sampling = launch_exp_largest_leveragescores_sampling(X_matrix,dataset_name,k,exp_number) 
error_fro_list_optimized_projection_DPP = launch_exp_optimized_projection_dpp(X_matrix,dataset_name,k,exp_number)    
error_fro_list_volume_sampling = launch_exp_volume_sampling(X_matrix,dataset_name,k,exp_number)    
error_fro_list_uniform_sampling = launch_exp_uniform_sampling(X_matrix,dataset_name,k,exp_number) 


#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP,error_fro_list_optimized_projection_DPP_with_rejection,error_fro_list_pivoted_qr], showfliers=False)
#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP,error_fro_list_optimized_projection_DPP_with_rejection,error_fro_list_pivoted_qr], showfliers=False)
#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_lvscores_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP], showfliers=False)
plt.figure(figsize=(10, 6)) 
ax = plt.subplot(111)   
#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP,error_fro_list_projection_DPP,error_fro_list_largest_lvs_sampling,error_fro_list_pivoted_qr_sampling,error_fro_list_derandomized_volume_sampling,error_fro_list_double_phase_sampling], showfliers=False)

error_fro_aggregated_list = []

error_fro_aggregated_list.append(error_fro_list_uniform_sampling)
error_fro_aggregated_list.append(error_fro_list_volume_sampling)
error_fro_aggregated_list.append(error_fro_list_optimized_projection_DPP)

plt.boxplot(error_fro_aggregated_list, showfliers=False)

#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP], showfliers=False)


plt.ylabel('$\|\mathrm{X- \pi_{S}(X)}\|_{Fr}$', fontsize=16)
 

#plt.legend(bbox_to_anchor=(1.04,1), loc="bottomleft", inset=.02, ["Volume Sampling","Projection DPP"])
#plt.legend("bottomleft", inset=.02, c("Volume Sampling","Projection DPP"), fill=topo.colors(3), horiz=TRUE, cex=0.8)
#plt.gca().xaxis.set_ticklabels(["Uniform Sampling","LVScores Sampling","Volume Sampling","Projection DPP"])

#plt.gca().xaxis.set_ticklabels(["Uniform Sampling","Volume Sampling","Optimized Projection DPP","Projection DPP","Largest lvs","Pivoted QR","derandomized_volume_sampling","Double Phase"])

plt.gca().xaxis.set_ticklabels(["Uniform Sampling","Volume Sampling","Projection DPP"])


plt.show()