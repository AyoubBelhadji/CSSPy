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

def plot_approximation_errors_on_toy_datasets(cardinal_global_list,cardinal_global_list_list,error_lists,legends_list,axislabel_list,filename_list):
    print(list(cardinal_global_list))
    #print(np.shape(list(error_lists[0])))
    palette_paired = plt.get_cmap('Paired')
    sim_number = len(cardinal_global_list)
    #plt.figure(figsize=(9, 6.5))
    #plt.scatter(cardinal_global_list,error_lists[0],label=legends_list[0],marker='_',color=palette_paired(1))
    #plt.bar(cardinal_global_list,list(error_lists[0]),label=legends_list[0],edgecolor=palette_paired(1),alpha=0)
    #for s_n in list(range(sim_number)):
    #    x = [-1.5, 0.72]
    #    y = [0.285987, 0.285987]
    #    ax.plot(x, y)
    x_index_min_bis = min(cardinal_global_list)
    x_index_max_bis = max(cardinal_global_list)
    delta_index = x_index_max_bis - x_index_min_bis
    error_lists_0 = error_lists[0][0:delta_index+1]
    error_lists_0.append(error_lists[0][delta_index])
    error_lists_0 = [error_lists[0][0]] + error_lists_0
    error_lists_3 = error_lists[3][0:delta_index+1]
    error_lists_3.append(error_lists[3][delta_index])
    error_lists_3 = [error_lists[3][0]] + error_lists_3
    cardinal_lists_0 = cardinal_global_list[0:delta_index+1]
    cardinal_lists_0.append(x_index_max_bis+0.5)
    cardinal_lists_0 = [x_index_min_bis-0.5]+cardinal_lists_0
    plt.step(cardinal_lists_0, error_lists_0, where='mid',label=legends_list[0],color=palette_paired(1))
    plt.scatter(cardinal_global_list_list,error_lists[1],label=legends_list[1],marker='x',color=palette_paired(0))
    plt.step(cardinal_lists_0, error_lists_3, where='mid',label=legends_list[3],color=palette_paired(5))
    print(error_lists[3])
    #plt.bar(cardinal_global_list,list(error_lists[3]),label=legends_list[3],edgecolor=palette_paired(5),alpha=0)
    plt.scatter(cardinal_global_list_list,error_lists[2],label=legends_list[2],marker='x',color=palette_paired(4))
    #plt.grid(True)
    #plt.gca().xaxis.grid(True, which='minor')  # minor grid on too
    #ax = plt.axes()
    #ax.xaxis.grid(True)
    x_index_min = min(cardinal_global_list_list)
    x_index_max = max(cardinal_global_list_list)
    plt.xlabel(axislabel_list[0], fontsize=11)
    plt.ylabel(r'$\mathrm{\mathbb{E} \|\| X- \pi_{C} X \|\| _{Fr}^{2}}$', fontsize=12)
    #plt.title('The case k = '+str(k)+', '+str(T)+' matrices')
    #plt.xticks(map(int, Y_cov[:-1]))
    plt.legend(bbox_to_anchor=(0.495,0.34), loc="upper left")
    plt.xticks(range(int(x_index_min[0]),int(x_index_max[0])+1, 1), fontsize=12)
    plt.tight_layout()
    figfile_title= filename_list[0]+filename_list[1]+"_matrices_number_"+filename_list[2]+"_N_"+filename_list[3]+"_"+filename_list[4]+".pdf"

    plt.savefig(figfile_title)
    plt.show()    
    
    

def plot_approximation_errors_effective_kernel_on_toy_datasets(cardinal_global_list,cardinal_global_list_list,error_lists,legends_list,axislabel_list,filename_list):
    print(list(cardinal_global_list))
    #print(np.shape(list(error_lists[0])))
    palette_paired = plt.get_cmap('Paired')
    sim_number = len(cardinal_global_list)
    #plt.figure(figsize=(9, 6.5))
    #plt.scatter(cardinal_global_list,error_lists[0],label=legends_list[0],marker='_',color=palette_paired(1))
    #plt.bar(cardinal_global_list,list(error_lists[0]),label=legends_list[0],edgecolor=palette_paired(1),alpha=0)
    #for s_n in list(range(sim_number)):
    #    x = [-1.5, 0.72]
    #    y = [0.285987, 0.285987]
    #    ax.plot(x, y)
    x_index_min_bis = min(cardinal_global_list)
    x_index_max_bis = max(cardinal_global_list)
    delta_index = x_index_max_bis - x_index_min_bis
    error_lists_0 = error_lists[0][0:delta_index+1]
    error_lists_0.append(error_lists[0][delta_index])
    error_lists_0 = [error_lists[0][0]] + error_lists_0
    error_lists_3 = error_lists[3][0:delta_index+1]
    error_lists_3.append(error_lists[3][delta_index])
    error_lists_3 = [error_lists[3][0]] + error_lists_3
    cardinal_lists_0 = cardinal_global_list[0:delta_index+1]
    cardinal_lists_0.append(x_index_max_bis+0.5)
    cardinal_lists_0 = [x_index_min_bis-0.5]+cardinal_lists_0
    plt.step(cardinal_lists_0, error_lists_0, where='mid',label=legends_list[0],color=palette_paired(1))
    plt.scatter(cardinal_global_list_list,error_lists[1],label=legends_list[1],marker='x',color=palette_paired(0))
    plt.step(cardinal_lists_0, error_lists_3, where='mid',label=legends_list[3],color=palette_paired(5))
    print(error_lists[3])
    #plt.bar(cardinal_global_list,list(error_lists[3]),label=legends_list[3],edgecolor=palette_paired(5),alpha=0)
    plt.scatter(cardinal_global_list_list,error_lists[2],label=legends_list[2],marker='x',color=palette_paired(4))
    #plt.grid(True)
    #plt.gca().xaxis.grid(True, which='minor')  # minor grid on too
    #ax = plt.axes()
    #ax.xaxis.grid(True)
    x_index_min = min(cardinal_global_list_list)
    x_index_max = max(cardinal_global_list_list)
    plt.xlabel(axislabel_list[0], fontsize=11)
    plt.ylabel(r'$\mathrm{\mathbb{E} \|\| X- \pi_{C} X \|\| _{Fr}^{2}}$', fontsize=12)
    #plt.title('The case k = '+str(k)+', '+str(T)+' matrices')
    #plt.xticks(map(int, Y_cov[:-1]))
    plt.legend(bbox_to_anchor=(0.495,0.34), loc="upper left")
    plt.xticks(range(int(x_index_min[0]),int(x_index_max[0])+1, 1), fontsize=12)
    plt.tight_layout()
    figfile_title= filename_list[0]+filename_list[1]+"_matrices_number_"+filename_list[2]+"_N_"+filename_list[3]+"_"+filename_list[4]+".pdf"
    
    plt.savefig(figfile_title)
    plt.show()    
    
    