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





def plot_results_of_multi_experiments(N,real_dim,r,T_,k_,mean,cov_,static_list_):
    print(np.diag(cov_))
    lv_scores_vector = k_/real_dim*np.ones(real_dim)  # The vector of leverage scores (the last one)
    T = deepcopy(T_) # The number of experiments
    versions_number = 1
    k = deepcopy(k_)
    cov_1 = deepcopy(cov_)
    volume_sampling_fro_list = []
    projection_dpp_fro_list = []
    p_eff_list = []
    cardinal_list = []
    avoiding_proba_list = []
    static_list = deepcopy(static_list_)
    volume_sampling_fro_list = []
    projection_dpp_fro_list = []
    #derandomized_projection_dpp_fro_list = []
    greedy_selection_fro_list = []
    effective_kernel_fro_list = []
    p_eff_list = []
    cardinal_list = []
    for t in range(T):
        print("t")
        print(t)
        #print("real_dim")
        #print(real_dim)
        random_cardinal_list = list(np.random.choice(static_list, 1))
        NAL_1 = Numrerical_Analysis_DPP(N,real_dim,r,k,versions_number,mean,cov_1,lv_scores_vector,random_cardinal_list)


        projection_DPP_res_fro_1 = NAL_1.get_expected_error_fro_for_projection_DPP()
        volume_sampling_res_fro_1 = NAL_1.get_expected_error_fro_for_volume_sampling()
        #derandomized_DPP_res_fro_1 = NAL_1.get_error_fro_for_derandomized_projection_DPP_selection()
        greedy_selection_res_fro_1 = NAL_1.get_error_fro_for_deterministic_selection()
        effective_kernel_sampling_res_fro_1 = NAL_1.get_expected_error_fro_for_effective_kernel_sampling()

    #    upper_tight_bound_projection_DPP_res_fro_1 = NAL_1.get_tight_upper_bound_error_fro_for_projection_DPP()

    #    alpha_sum_res_1 = NAL_1.get_alpha_sum_k_leverage_scores(1)


    #    sum_U_res_1 = NAL_1.get_sum_k_leverage_scores()
        p_eff_res_1 = NAL_1.get_p_eff_leverage_scores()
        
        
        avoiding_proba_res_1 = NAL_1.get_avoiding_probability()
        avoiding_proba_list.append(avoiding_proba_res_1)
        greedy_selection_fro_list.append(greedy_selection_res_fro_1)
        #derandomized_projection_dpp_fro_list.append(derandomized_DPP_res_fro_1)
        
        effective_kernel_fro_list.append(list(effective_kernel_sampling_res_fro_1))

        volume_sampling_fro_list.append(list(volume_sampling_res_fro_1))
        projection_dpp_fro_list.append(list(projection_DPP_res_fro_1))
        p_eff_list.append(list(p_eff_res_1))
        cardinal_list.append(random_cardinal_list) 
        print("next")
    flattened_cardinal_list= [item for items in cardinal_list for item in items]
    flattened_p_eff_list= [item for items in p_eff_list for item in items]

    theoretical_projection_DPP_error_bound_list = from_p_eff_to_error_bound(flattened_cardinal_list,k,real_dim)


    plt.scatter(cardinal_list,projection_dpp_fro_list,label="Projection DPP Sampling",marker='_')
    plt.scatter(cardinal_list,volume_sampling_fro_list,label="Volume Sampling",marker='_')
    #plt.scatter(cardinal_list,derandomized_projection_dpp_fro_list,label="derandomized projection dpp", marker='_')
    plt.scatter(cardinal_list,greedy_selection_fro_list,label = "greedy", marker='_',color = 'purple')
    plt.scatter(cardinal_list,theoretical_projection_DPP_error_bound_list,color='red',marker='_',label="Theoretical bound for Projection DPP Sampling")

    plt.xlabel(r'$p$', fontsize=12)
    plt.ylabel(r'$\mathbb{E}_{S \sim \mathcal{P}}(\|X- \pi_{S}(X)\|_{Fr}^{2})$', fontsize=12)
    plt.title('The case k = '+str(k)+', '+str(T)+' matrices, flat spectrum after k+1')
    #plt.xticks(map(int, Y_cov[:-1]))

    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
    theoretical_effective_kernel_error_bound_list = from_p_eff_to_error_bound_2(flattened_p_eff_list,k,real_dim)


    #theoretical_effective_kernel_error_bound_list = from_p_eff_to_error_bound(flattened_p_eff_list,k,real_dim)


    plt.scatter(p_eff_list,effective_kernel_fro_list,label="Effective Kernel Sampling",marker='_')
    plt.scatter(p_eff_list,volume_sampling_fro_list,label="Volume Sampling",marker='_')
    #plt.scatter(p_eff_list,derandomized_projection_dpp_fro_list,label="derandomized projection dpp", marker='_')
    plt.scatter(p_eff_list,theoretical_effective_kernel_error_bound_list,color='red',marker='_',label="Theoretical bound for Effective Kernel Sampling")
    plt.scatter(p_eff_list,greedy_selection_fro_list,label = "greedy", marker='_',color = 'purple')
    plt.xlabel(r'$p_{eff}(\frac{1}{2})$', fontsize=12)
    plt.ylabel(r'$\mathrm{\mathbb{E}_{S \sim \mathcal{P}}(\|X- \pi_{S}(X)\|_{Fr}^{2})$', fontsize=12)
    plt.title('The case k = '+str(k)+', '+str(T)+' matrices, flat spectrum after k+1')


    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
    
    
    
    
    
    
    plt.scatter(cardinal_list,projection_dpp_fro_list,label="Projection DPP Sampling",marker='_')
    plt.scatter(cardinal_list,volume_sampling_fro_list,label="Volume Sampling",marker='_')
    #plt.scatter(cardinal_list,derandomized_projection_dpp_fro_list,label="derandomized projection dpp", marker='_')
    plt.scatter(cardinal_list,theoretical_projection_DPP_error_bound_list,color='red',marker='_',label="Theoretical bound for Projection DPP Sampling")

    plt.xlabel(r'$p$', fontsize=12)
    plt.ylabel(r'$\mathbb{E}_{S \sim \mathcal{P}}(\|X- \pi_{S}(X)\|_{Fr}^{2})$', fontsize=12)
    plt.title('The case k = '+str(k)+', '+str(T)+' matrices, flat spectrum after k+1')
    #plt.xticks(map(int, Y_cov[:-1]))

    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
    
    
    
    plt.scatter(p_eff_list,effective_kernel_fro_list,label="Effective Kernel Sampling",marker='_')
    plt.scatter(p_eff_list,volume_sampling_fro_list,label="Volume Sampling",marker='_')
    #plt.scatter(p_eff_list,derandomized_projection_dpp_fro_list,label="derandomized projection dpp", marker='_')
    plt.scatter(p_eff_list,theoretical_effective_kernel_error_bound_list,color='red',marker='_',label="Theoretical bound for Effective Kernel Sampling")
    plt.xlabel(r'$p_{eff}(\frac{1}{2})$', fontsize=12)
    plt.ylabel(r'$\mathbb{E}_{S \sim \mathcal{P}}(\|X- \pi_{S}(X)\|_{Fr}^{2})$', fontsize=12)
    plt.title('The case k = '+str(k)+', '+str(T)+' matrices, flat spectrum after k+1')


    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
    
    plt.scatter(p_eff_list,avoiding_proba_list,label="Avoiding Probability")

    plt.xlabel(r'$p_{eff}(\frac{1}{2})$', fontsize=12)
    plt.ylabel(r'$\mathbb{P}(S\cap T_{eff} = \emptyset)$', fontsize=12)
    plt.title('The case k = '+str(k)+', '+str(T)+' matrices, flat spectrum after k+1')


    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
    print("N")
    print(N)