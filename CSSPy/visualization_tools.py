import sys
#sys.path.append('../')
sys.path.insert(0, '..')

import numpy as np

from matplotlib import pyplot as plt

def plot_leverage_scores(klv_vector,dataset_name,k):
    """ Plot k-leverage scores profile.
    :param klv_vector: 
        The sorted vector of k-leverage scores of a matrix.
    :type klv_vector: 
        array_type
    :param dataset_name:
        The name of the dataset.
    :type dataset_name:
        string
    :param k: 
        The order of the low rank apparoximation.
    :type k: 
        int
    :return: 
        A plot of the sorted vector of k-leverage scores of the dataset.
    :rtype: 
        void
    """
    [N] = np.shape(klv_vector)
    plt.plot(list(range(N)), klv_vector, color="#3F5D7D")
    plt.xlabel("i", fontsize=19)
    plt.ylabel(r'$\mathrm{\ell}_{i}^{k}$', fontsize=19)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    figfile_name= dataset_name+"_klv_k_"+str(k)+".pdf"
    plt.savefig(figfile_name)
    plt.show()


def plot_cumul_leverage_scores(klv_vector,dataset_name,k):
    """ Plot k-leverage scores profile and cumulative k-leverage scores.
    :param klv_vector: 
        The sorted vector of k-leverage scores of a matrix.
    :type klv_vector: 
        array_type
    :param dataset_name:
        The name of the dataset.
    :type dataset_name:
        string
    :param k: 
        The order of the low rank apparoximation.
    :type k: 
        int
    :return: 
        A plot of the sorted vector of k-leverage scores of the dataset as well as the cumulative k-leverage scores.
    :rtype: 
        void
    """
    [N] = np.shape(klv_vector)
    cumul_klv_vector = np.cumsum(klv_vector)
    fig, ax1 = plt.subplots()
    x_axis_list = list(range(N))
    lns1 = ax1.plot(x_axis_list, cumul_klv_vector, color="#3F5D7D", label = "Cumulative k-leverage scores")
    #ax1.fill_between(x_axis_list,[0]*N,klv_vector, alpha=0.7)
    ax2 = ax1.twinx()
    lns2 = ax2.plot(x_axis_list, klv_vector, color="red", label = "k-leverage scores") 
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="lower right", bbox_to_anchor=(0.95,0.05))
    ax1.set_xlabel("i",fontsize=16)
    ax1.set_ylabel(r'$\sum_{j=1}^{i}\mathrm{\ell}_{j}^{k}$',color="#3F5D7D")
    ax1.tick_params(axis='y',labelcolor="#3F5D7D")
    ax2.set_ylabel(r'$\mathrm{\ell}_{i}^{k}$',color="red")
    ax2.tick_params(colors='red')
    plt.xticks(fontsize=16)
    figfile_name= dataset_name+"_cumul_klv_k_"+str(k)+".pdf"
    plt.savefig(figfile_name)
    plt.show()



def plot_cumul_leverage_scores_2(klv_vector,dataset_name,k):
    """ Plot k-leverage scores profile and cumulative k-leverage scores.
    :param klv_vector: 
        The sorted vector of k-leverage scores of a matrix.
    :type klv_vector: 
        array_type
    :param dataset_name:
        The name of the dataset.
    :type dataset_name:
        string
    :param k: 
        The order of the low rank apparoximation.
    :type k: 
        int
    :return: 
        A plot of the sorted vector of k-leverage scores of the dataset as well as the cumulative k-leverage scores.
    :rtype: 
        void
    """
    [N] = np.shape(klv_vector)
    cumul_klv_vector = np.cumsum(klv_vector)
    print(klv_vector)
    plt.figure(figsize=(10, 6)) 
    fig, ax1 = plt.subplots()
    x_axis_list = list(range(N))
    plt.plot(x_axis_list, cumul_klv_vector, color="#3F5D7D")
    plt.plot(x_axis_list, 10*klv_vector, color="r") 
    ax = plt.subplot(111)        
    ax.yaxis.grid()
    ax.spines["top"].set_visible(False)       
    ax.spines["right"].set_visible(False)    
    plt.xlabel("i", fontsize=19)
    plt.ylabel(r'$\sum_{j=1}^{i}\mathrm{\ell}_{j}^{k}$', fontsize=19)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    figfile_name= dataset_name+"_cumul_klv_k_"+str(k)+".pdf"
    plt.savefig(figfile_name)
    plt.show()