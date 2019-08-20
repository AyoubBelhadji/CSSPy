import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import binom
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, HTML

#sys.path.append("../")
from FrameBuilder.eigenstepsbuilder import *
from decimal import *
from copy import deepcopy
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from env.numerical_analysis_dpp import *
from env.plot_functions import *


def swap_elements_of_list(list_1,indices):
    list_1_ = deepcopy(list_1)
    N = len(list_1_)
    list_indices = []
    #print(N)
    for item in indices:
        #print(item)
        list_indices.append(list_1_[item])
    list_1_2 = list_indices #+ list_1_ 
    #list_final = list_1_2[0:N]
    return list_1_2


def extract_first_elements(list_1):


    list_1_ = deepcopy(list_1)
    max_list_1 = max(list_1_)
    min_list_1 = min(list_1_)
    effective_number = max_list_1 - min_list_1+1
    N = len(list_1_)
    index_of_firsts = [0]*effective_number
    index_of_firsts[0] = 0
    counter = 1
    counter_on_N = 1
    hit_indices = [list_1_[0]]
    #print(effective_number)
    #print(list_1_)
    #print(list_2_)
    while counter<effective_number:
        if not(list_1_[counter_on_N] in hit_indices):
            index_of_firsts[counter] = counter_on_N
            counter = counter +1
            hit_indices.append(list_1_[counter_on_N])
        counter_on_N = counter_on_N +1
    #print(index_of_firsts)
    list_2_ = [list_1_[i] for i in index_of_firsts]
    I_arg_sort = np.argsort(list_2_)
    list_3_ = []
    sorted_index_of_firsts = []
    for i in I_arg_sort:
        sorted_index_of_firsts.append(index_of_firsts[i])
        list_3_.append(list_1_[index_of_firsts[i]])
    return list_3_,sorted_index_of_firsts
        
#
#def extract_first_elements(list_1,list_2):
#
#
#    list_1_ = deepcopy(list_1)
#    list_2_ = deepcopy(list_2)
#    max_list_1 = max(list_1_)
#    min_list_1 = min(list_1_)
#    
#    array = np.array(list_1_)
#    effective_number = max_list_1 - min_list_1
#    N = len(list_1_)
#    index_of_firsts = [0]*effective_number
#    index_of_firsts[0] = 0
#    counter = 1
#    counter_on_N = 1
#    while counter<effective_number-2:
#        if list_1_[counter_on_N] != list_1_[counter_on_N-1]:
#            index_of_firsts[counter] = counter_on_N
#            counter = counter +1
#        counter_on_N = counter_on_N +1
#    return [list_1_[i] for i in index_of_firsts],[list_2_[i] for i in index_of_firsts]
#        
        
def generate_list_of_list_from_list(list_1):
    list_1_ = deepcopy(list_1)
    list_of_list = []
    for item in list_1_:
        list_of_list.append([item])
    return list_of_list

def plot_results_of_multi_experiments(N,real_dim,r,T_,k_,mean,cov_,static_list_,activate_correction_factor,file_name_comment):
    #print(np.diag(cov_))
    lv_scores_vector = k_/real_dim*np.ones(real_dim)  # The vector of leverage scores (the last one)
    T = deepcopy(T_) # The number of experiments
    versions_number = 1
    epsilon_vizualisation = 0.01
    k = deepcopy(k_)
    cov_1 = deepcopy(cov_)
    volume_sampling_fro_list = []
    projection_dpp_fro_list = []
    p_eff_list = []
    cardinal_list = []
    cardinal_global_list_list = []
    avoiding_proba_list = []
    static_list = deepcopy(static_list_)
    volume_sampling_fro_list = []
    projection_dpp_fro_list = []
    #derandomized_projection_dpp_fro_list = []
    greedy_selection_fro_list = []
    effective_kernel_fro_list = []
    p_eff_list = []
    p_eff_list_list = []
    cardinal_global_list = []
    theta_list = []
    theta_complete_list = []
    theoretical_bound_avoiding_probability_list = []
    static_list_len = len(static_list)
    static_list_counter = 0
    matrix_rank = min(np.count_nonzero(cov_),N)
    correction_factor = 1
    if activate_correction_factor == 1:
        beta_factor = cov_[k,k]/cov_[matrix_rank-1,matrix_rank-1]
        dimension_factor = (real_dim - k_)/(matrix_rank - k_)
        correction_factor = np.float(beta_factor)**2*np.float(dimension_factor)
        
    for t in range(T):
        print("Matrix number")
        print(t)
        #print(correction_factor)
        #print("real_dim")
        #print(real_dim)
        cardinal_list_element = static_list[static_list_counter] #list(np.random.choice(static_list, 1))
        cardinal_list = [static_list[static_list_counter]] #list(np.random.choice(static_list, 1))
        static_list_counter = static_list_counter +1
        if static_list_counter == static_list_len:
            static_list_counter = 0
        NAL_1 = Numrerical_Analysis_DPP(N,real_dim,r,k,versions_number,mean,cov_1,lv_scores_vector,cardinal_list)
        #print("NAL")

        projection_DPP_res_fro_1 = (1-epsilon_vizualisation)*NAL_1.get_expected_error_fro_for_projection_DPP()
        volume_sampling_res_fro_1 = (1-epsilon_vizualisation)*NAL_1.get_expected_error_fro_for_volume_sampling()
        #derandomized_DPP_res_fro_1 = NAL_1.get_error_fro_for_derandomized_projection_DPP_selection()
        greedy_selection_res_fro_1 = NAL_1.get_error_fro_for_deterministic_selection()
        effective_kernel_sampling_res_fro_1 = NAL_1.get_expected_error_fro_for_effective_kernel_sampling()

    #    upper_tight_bound_projection_DPP_res_fro_1 = NAL_1.get_tight_upper_bound_error_fro_for_projection_DPP()

    #    alpha_sum_res_1 = NAL_1.get_alpha_sum_k_leverage_scores(1)


    #    sum_U_res_1 = NAL_1.get_sum_k_leverage_scores()
        p_eff_res_1 = NAL_1.get_p_eff_leverage_scores()
        
        
        avoiding_proba_res_1,theta_list,avoiding_proba_theoretical_list = NAL_1.get_avoiding_probability()
        avoiding_proba_list.append(avoiding_proba_res_1)
        greedy_selection_fro_list.append(greedy_selection_res_fro_1)
        theta_complete_list.append(theta_list)
        #theoretical_bound_avoiding_probability_list.append(avoiding_proba_theoretical_list)
        #derandomized_projection_dpp_fro_list.append(derandomized_DPP_res_fro_1)
        
        effective_kernel_fro_list.append(list(effective_kernel_sampling_res_fro_1))

        volume_sampling_fro_list.append(list(volume_sampling_res_fro_1))
        projection_dpp_fro_list.append(list(projection_DPP_res_fro_1))
        p_eff_list_list.append(list(p_eff_res_1))
        p_eff_list_element = int(p_eff_res_1[0])
        p_eff_list.append(p_eff_list_element)
        cardinal_global_list.append(cardinal_list_element) 
        cardinal_global_list_list.append(cardinal_list)
        #print("next")
    for theta in theta_list:
        theoretical_bound_avoiding_probability_list.append(1/theta)
    #avoiding_proba_list,theta_list = NAL_1.get_avoiding_probability()
    #versions_number = int(len(avoiding_proba_list)/len(theta_list))
    #ones_list = [1]*versions_number
    #theta_complete_list = list(np.kron(ones_list,theta_list))

    flattened_cardinal_list= [item for items in cardinal_global_list_list for item in items]
    flattened_p_eff_list= [item for items in p_eff_list_list for item in items]

    theoretical_projection_DPP_error_bound_list_pre_factor = from_p_eff_to_error_bound(flattened_cardinal_list,k,real_dim)
    theoretical_projection_DPP_error_bound_list = [correction_factor * i for i in theoretical_projection_DPP_error_bound_list_pre_factor]
    
    theoretical_effective_kernel_error_bound_list_pre_factor = from_p_eff_to_error_bound_2(flattened_p_eff_list,k,real_dim)
    theoretical_effective_kernel_error_bound_list = [correction_factor * i for i in theoretical_effective_kernel_error_bound_list_pre_factor]
    cardinal_global_list_len = len(cardinal_global_list_list)
    volume_sampling_fro_bound_list = [k+1]*cardinal_global_list_len
    error_lists = []
    error_lists.append(volume_sampling_fro_bound_list)
    error_lists.append(volume_sampling_fro_list)
    error_lists.append(projection_dpp_fro_list)
    error_lists.append(theoretical_projection_DPP_error_bound_list)
    legends_list = []
    legends_list.append("Borne th. VS")
    legends_list.append("VS")
    legends_list.append("PPD")
    legends_list.append("Borne th. PPD")
    axislabel_list = []
    axislabel_list.append(r'$\mathrm{p}$')
    filename_list = []
    filename_list.append("dpp_k_")
    filename_list.append(str(k))
    filename_list.append(str(T))
    filename_list.append(str(N))
    filename_list.append(file_name_comment)
    
    plot_approximation_errors_on_toy_datasets(cardinal_global_list,cardinal_global_list_list,error_lists,legends_list,axislabel_list,filename_list)
    
#    palette_paired = plt.get_cmap('Paired')
#    #palette_PuBuGn = plt.get_cmap('PuBuGn')
#
#    
#    plt.scatter(cardinal_global_list,volume_sampling_fro_bound_list,label="Volume sampling bound",marker='_',color=palette_paired(1))
#    plt.scatter(cardinal_global_list,volume_sampling_fro_list,label="Volume sampling",marker='_',color=palette_paired(0))
#    plt.scatter(cardinal_global_list,projection_dpp_fro_list,label="Projection DPP",marker='_',color=palette_paired(4))
#    #plt.scatter(cardinal_list,derandomized_projection_dpp_fro_list,label="derandomized projection dpp", marker='_')
#    plt.scatter(cardinal_global_list,theoretical_projection_DPP_error_bound_list,marker='_',label="Projection DPP bound",color=palette_paired(5))
#    plt.xlabel(r'$\mathrm{p}$', fontsize=12)
#    plt.ylabel(r'$\mathrm{\mathbb{E} \|\| X- \pi_{C} X \|\| _{Fr}^{2}}$', fontsize=12)
#    plt.title('The case k = '+str(k)+', '+str(T)+' matrices')
#    #plt.xticks(map(int, Y_cov[:-1]))
#    plt.legend(bbox_to_anchor=(0.495,0.34), loc="upper left")
#    plt.xticks(range(4, 21, 1), fontsize=12)
#    figfile_title= "dpp_k_"+str(k)+"_matrices_number_"+str(T)+"_N_"+str(N)+"_"+file_name_comment+".pdf"
#    plt.savefig(figfile_title)
#    plt.show()
    #####
    #####
    #####
    legends_list = []    
    legends_list.append("V.S. bound")
    legends_list.append("V.S.")
    legends_list.append("R.P. DPP")
    legends_list.append("R.P. DPP")
    error_lists = []
    axislabel_list = []
    axislabel_list.append(r'$\mathrm{p_{eff}}(\frac{1}{2})$')
    #print(np.shape(volume_sampling_fro_bound_list))
    #print(p_eff_list)
    #print(error_lists[0])
    #print(volume_sampling_fro_bound_list)
    p_eff_list_len = len(p_eff_list)
    
    #error_list_len = len(error_lists[0])
    p_eff_list_temp,indices_list = extract_first_elements(p_eff_list)
    p_eff_list = swap_elements_of_list(p_eff_list,indices_list)
    #print(p_eff_list)
    #p_eff_list = p_eff_list_temp + p_eff_list 
    #p_eff_list = p_eff_list[0:error_list_len-p_eff_list_len]
    #print(len(p_eff_list))
    p_eff_list_list_temp = generate_list_of_list_from_list(p_eff_list)
    #p_eff_list_list = p_eff_list_list_temp + p_eff_list_list
    volume_sampling_fro_bound_list_ = swap_elements_of_list(volume_sampling_fro_bound_list,indices_list)
    theoretical_effective_kernel_error_bound_list_ = swap_elements_of_list(theoretical_effective_kernel_error_bound_list,indices_list)
    
    error_lists.append(volume_sampling_fro_bound_list_)
    error_lists.append(volume_sampling_fro_list)
    error_lists.append(effective_kernel_fro_list)
    error_lists.append(theoretical_effective_kernel_error_bound_list_)
    filename_list = []
    filename_list.append("effective_kernel_k_")
    filename_list.append(str(k))
    filename_list.append(str(T))
    filename_list.append(str(N))
    filename_list.append(file_name_comment)


    plot_approximation_errors_effective_kernel_on_toy_datasets(p_eff_list,p_eff_list_list,error_lists,legends_list,axislabel_list,filename_list)
    
#    plt.scatter(p_eff_list,volume_sampling_fro_bound_list,label="Volume sampling bound",marker='_',color=palette_paired(1))
#    plt.scatter(p_eff_list,volume_sampling_fro_list,label="Volume Sampling",marker='_',color=palette_paired(0))
#    #plt.scatter(p_eff_list,derandomized_projection_dpp_fro_list,label="derandomized projection dpp", marker='_')
#    plt.scatter(p_eff_list,effective_kernel_fro_list,label="Effective kernel",marker='_',color=palette_paired(4))
#    plt.scatter(p_eff_list,theoretical_effective_kernel_error_bound_list,marker='_',label="Effective kernel bound",color=palette_paired(5))
#    plt.xlabel(r'$\mathrm{p_{eff}(\frac{1}{2})}$', fontsize=12)
#    plt.ylabel(r'$\mathrm{\mathbb{E} \|\| X- \pi_{C} X \|\| _{Fr}^{2}}$', fontsize=12)
#    plt.title('The case k = '+str(k)+', '+str(T)+' matrices')
#    plt.legend(bbox_to_anchor=(0.495,0.34), loc="upper left")
#    plt.xticks(range(2, 13, 1), fontsize=12)
#    figfile_title= "effective_kernel_k_"+str(k)+"_matrices_number_"+str(T)+"_N_"+str(N)+"_"+file_name_comment+".pdf"
#    plt.savefig(figfile_title)
#    plt.show()
    #####
    #####
    #####

    plt.scatter(theta_complete_list,avoiding_proba_list,label="Avoiding Probability",marker='x')

    
    plt.plot(theta_list,theoretical_bound_avoiding_probability_list,color='red',label="Theoretical bound")#)
    plt.xlabel(r'$\mathrm{\theta}$', fontsize=16)
    plt.ylabel(r'$\mathrm{\mathbb{P}(S\cap T_{eff} = \emptyset)}$', fontsize=16)
    #plt.title('The case k = '+str(k)+', '+str(T)+' matrices')
    plt.legend(bbox_to_anchor=(0.55,1), loc="upper left")
    plt.xticks(fontsize=12)
    #plt.tight_layout()
    figfile_title= "avoid_proba_k_"+str(k)+"_matrices_number_"+str(T)+"_N_"+str(N)+"_"+file_name_comment+".pdf"
    plt.savefig(figfile_title)
    plt.show()

