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
dataset_name = "RELATHE"
dataset_file = "RELATHE"+str("_X")
clustering_labels_file = "RELATHE"+str("_Y")
t = timeit.Timer('char in text', setup='text = "sample string"; char = "g"')
#X_df = pd.read_csv('datasets/'+dataset_file+'.csv', sep=" ", header=None)

X_df = pd.read_csv('datasets/arcene_train.data', header=None, delimiter=' ', skip_blank_lines=True)
Y_df = pd.read_csv('datasets/arcene_train.labels', sep=" ", header=None)

X_matrix = X_df.values
Y_matrix = (-Y_df.values +1)/2
Y_matrix_2 = (Y_df.values +1)/2
#mean_X = X_matrix.mean(axis=0)
#X_matrix = X_matrix - mean_X
print(np.shape(X_matrix))
d = np.shape(X_matrix)[1]
N = np.shape(X_matrix)[0] -1
t_1 = timeit.default_timer() 

#_,D,V = np.linalg.svd(X_matrix)

U,D,V = svds(X_matrix, 61 , return_singular_vectors='true')


k_means_test = k_means_using_column_subset(2,U[:,0:2])
print("kmeans")
print(k_means_test)
print(evaluate_k_means_using_jaccard(list(Y_matrix[:,0]),list(k_means_test)))
#print(evaluate_k_means_using_jaccard(list(Y_matrix_2[:,0]),list(k_means_test)))

print("finish kmeans")
k = 2
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

exp_number = 50

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
#error_fro_list_optimized_projection_DPP = launch_exp_optimized_projection_dpp(X_matrix,dataset_name,k,exp_number)    
error_fro_list_volume_sampling = launch_exp_volume_sampling(X_matrix,Y_matrix,dataset_name,k,exp_number)    
error_fro_list_uniform_sampling = launch_exp_uniform_sampling(X_matrix,dataset_name,k,exp_number) 


#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP,error_fro_list_optimized_projection_DPP_with_rejection,error_fro_list_pivoted_qr], showfliers=False)
#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP,error_fro_list_optimized_projection_DPP_with_rejection,error_fro_list_pivoted_qr], showfliers=False)
#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_lvscores_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP], showfliers=False)
  
#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP,error_fro_list_projection_DPP,error_fro_list_largest_lvs_sampling,error_fro_list_pivoted_qr_sampling,error_fro_list_derandomized_volume_sampling,error_fro_list_double_phase_sampling], showfliers=False)


def random_error_list_to_min_error_list(error_list):
    l_test = len(error_list)
    min_value_random_list = min(error_list)
    new_list = [min_value_random_list]*l_test
    return new_list


error_fro_aggregated_list = []

boosting_error_fro_aggregated_list = []

min_error_fro_list_uniform_sampling = random_error_list_to_min_error_list(error_fro_list_uniform_sampling)
min_error_fro_list_volume_sampling = random_error_list_to_min_error_list(error_fro_list_volume_sampling)
min_error_fro_list_optimized_projection_DPP = random_error_list_to_min_error_list(error_fro_list_optimized_projection_DPP)
min_error_fro_list_double_phase_sampling = random_error_list_to_min_error_list(error_fro_list_double_phase_sampling)




boosting_error_fro_aggregated_list.append(min_error_fro_list_uniform_sampling)
boosting_error_fro_aggregated_list.append(min_error_fro_list_volume_sampling)
boosting_error_fro_aggregated_list.append(min_error_fro_list_optimized_projection_DPP)
boosting_error_fro_aggregated_list.append(error_fro_list_largest_lvs_sampling)
boosting_error_fro_aggregated_list.append(error_fro_list_pivoted_qr_sampling)
boosting_error_fro_aggregated_list.append(min_error_fro_list_double_phase_sampling)




error_fro_aggregated_list.append(error_fro_list_uniform_sampling)
error_fro_aggregated_list.append(error_fro_list_volume_sampling)
error_fro_aggregated_list.append(error_fro_list_optimized_projection_DPP)
error_fro_aggregated_list.append(error_fro_list_largest_lvs_sampling)
error_fro_aggregated_list.append(error_fro_list_pivoted_qr_sampling)
error_fro_aggregated_list.append(error_fro_list_double_phase_sampling)

plt.figure(figsize=(10, 6)) 
ax = plt.subplot(111) 
plt.boxplot(error_fro_aggregated_list, showfliers=False)

#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP], showfliers=False)

plt.ylabel('$\|\mathrm{X- \pi_{S}(X)}\|_{Fr}$', fontsize=16)
 
#plt.legend(bbox_to_anchor=(1.04,1), loc="bottomleft", inset=.02, ["Volume Sampling","Projection DPP"])
#plt.legend("bottomleft", inset=.02, c("Volume Sampling","Projection DPP"), fill=topo.colors(3), horiz=TRUE, cex=0.8)
#plt.gca().xaxis.set_ticklabels(["Uniform Sampling","LVScores Sampling","Volume Sampling","Projection DPP"])

#plt.gca().xaxis.set_ticklabels(["Uniform Sampling","Volume Sampling","Optimized Projection DPP","Projection DPP","Largest lvs","Pivoted QR","derandomized_volume_sampling","Double Phase"])

plt.gca().xaxis.set_ticklabels(["Uniform Sampling","Volume Sampling","Projection DPP","Largest lvs","Pivoted QR","Double Phase"])


plt.show()



plt.figure(figsize=(10, 6)) 
ax = plt.subplot(111) 
plt.boxplot(boosting_error_fro_aggregated_list, showfliers=False)

#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP], showfliers=False)

plt.ylabel('$\|\mathrm{X- \pi_{S}(X)}\|_{Fr}$', fontsize=16)
 
#plt.legend(bbox_to_anchor=(1.04,1), loc="bottomleft", inset=.02, ["Volume Sampling","Projection DPP"])
#plt.legend("bottomleft", inset=.02, c("Volume Sampling","Projection DPP"), fill=topo.colors(3), horiz=TRUE, cex=0.8)
#plt.gca().xaxis.set_ticklabels(["Uniform Sampling","LVScores Sampling","Volume Sampling","Projection DPP"])

#plt.gca().xaxis.set_ticklabels(["Uniform Sampling","Volume Sampling","Optimized Projection DPP","Projection DPP","Largest lvs","Pivoted QR","derandomized_volume_sampling","Double Phase"])

plt.gca().xaxis.set_ticklabels(["Uniform Sampling","Volume Sampling","Projection DPP","Largest lvs","Pivoted QR","Double Phase"])


plt.show()

from astropy.io import ascii




def plot_CSSP_dim_2_with_labels():
    
#    X_df = pd.read_csv('datasets/'+dataset_name+'_X.csv', sep=",", header=None)
#    Y_df = pd.read_csv('datasets/'+dataset_name+'_Y.csv', sep=",", header=None)
    #X_df = pd.read_csv('datasets/arcene_train.data', header=None, delimiter=' ', skip_blank_lines=True)
    #Y_df = pd.read_csv('datasets/arcene_train.labels', sep=" ", header=None)
    


    #data = ascii.read("datasets/vectors.dat")
    #df = df.fillna(0)
    
    #df_2 = df.iloc[:,0:4026]
    #X_matrix = df_2.values    
    #Y_df = df.iloc[:,4026]


    words_df = pd.read_csv('datasets/features.idx', sep=" ", header=None)
    
    words_df_2 = words_df.iloc[:,1]
    N = 17515
    admissible_words = []
    for l in range(N):
        if len(str(words_df_2[l]))> 6:
            admissible_words.append(l)
    X_matrix = X[:,admissible_words]
    Y_matrix = y
    #print(Y_df)
    #mean_X = X_matrix.mean(axis=0)
    #X_matrix = X_matrix - mean_X
    N = np.shape(X_matrix)[0]
    Y_array = [0]*N
    Y_p = []
    Y_n = []
    U,S,V = svds(X_matrix, 2 , return_singular_vectors='true')

    print(np.shape(X_matrix))
    V_k = calculate_right_eigenvectors_k_svd(X_matrix,15)
    
    d = np.shape(X_matrix)[1]
    #projection_DPP_Sampler(X_matrix, k, V_k,d)

        
    NAL = projection_DPP_Sampler(X_matrix, 15, V_k,d)
    A_S = NAL.MultiRounds()
    print(A_S)
    #_,D,V = np.linalg.svd(X_matrix)
    #selected_columns_indices = [58, 117, 217, 269, 274, 314, 418, 630, 739, 927, 969, 993, 1459, 1624, 1651]
    #selected_columns_indices = [14, 52, 217, 555, 675, 694, 746, 781, 985, 1342, 1510, 1511, 1596, 1624, 1862]
    selected_columns_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    #selected_columns_indices = [110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]
    columns_of_X = X_matrix[:,selected_columns_indices]
    print(np.shape(columns_of_X))
    U,S,V = svds(A_S, 2 , return_singular_vectors='true')
    for l in range(N):
        print(int(Y_matrix[l]))
        if U[l,1] < 0.3 and U[l,0] < 0.4 :
            Y_array[l] = int(Y_matrix[l])
            
            if int(Y_matrix[l])>0:
                 Y_p.append(l)
            else:
                Y_n.append(l)
    #np.linalg.svd(X_matrix, 2)
    #Y_p = np.where(Y_array == int(1))
    #Y_n = np.where(Y_array == int(-1))
    print(S)
    plt.scatter(U[Y_p,0], U[Y_p,1])
    plt.scatter(U[Y_n,0], U[Y_n,1])
    plt.show()
    
#from scipy.io import arff
#import pandas as pd

#data = arff.loadarff('datasets/lymphoma.arff.txt')
#df = pd.DataFrame(data[0])

#df = df.fillna(0)

#df_2 = df.iloc[:,0:4026]
#df_matrix = df_2.values

#df.head()


from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
mem = Memory("./mycache")

@mem.cache
def get_data():
    data = load_svmlight_file("datasets/vectors.dat")
    return data[0].todense(), data[1]

X, y = get_data()

