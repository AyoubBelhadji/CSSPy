import sys
#sys.path.append('../')
sys.path.insert(0, '..')

import numpy as np

from matplotlib import pyplot as plt

def plot_leverage_scores(klv_vector,dataset_name,k):
    [N] = np.shape(klv_vector)
    plt.figure(figsize=(10, 6)) 
    plt.plot(list(range(N)), klv_vector, color="#3F5D7D")
    plt.xlabel("i", fontsize=19)
    plt.ylabel(r'$\mathrm{\ell}_{i}^{k}$', fontsize=19)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    #plt.title("The k-leverage scores of the matrix "+dataset_name+" (k = "+str(k)+")")
    figfile_name= dataset_name+"_klv_k_"+str(k)+".pdf"
    plt.savefig(figfile_name)
    plt.show()


def plot_cumul_leverage_scores(klv_vector,dataset_name,k):
    [N] = np.shape(klv_vector)
    cumul_klv_vector = np.cumsum(klv_vector)
    print(klv_vector)
    #plt.figure(figsize=(11, 6)) 
    fig, ax1 = plt.subplots()
    x_axis_list = list(range(N))
    ax1.plot(x_axis_list, cumul_klv_vector, color="#3F5D7D")
    ax1.fill_between(x_axis_list,[0]*N,cumul_klv_vector, alpha=0.7)
    #ax_1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    ax2.plot(x_axis_list, klv_vector, color="red") 
#    ax = plt.subplot(111)   
#    #ax = plt.axes()        
#    ax.yaxis.grid()
#    ax.spines["top"].set_visible(False)    
#    #ax.spines["bottom"].set_visible(False)    
#    ax.spines["right"].set_visible(False)    
    #ax.spines["left"].set_visible(False) 
    ax1.set_xlabel("i")
    ax1.set_ylabel(r'$\sum_{j=1}^{i}\mathrm{\ell}_{j}^{k}$')
    ax2.set_ylabel(r'$\mathrm{\ell}_{i}^{k}$')
#        
#    plt.xticks(fontsize=16)
#    plt.yticks(fontsize=16)
        
    #plt.title("The k-leverage scores of the matrix "+dataset_name+" (k = "+str(k)+")")
    figfile_name= dataset_name+"_cumul_klv_k_"+str(k)+".pdf"
    plt.savefig(figfile_name)
    plt.show()



def plot_cumul_leverage_scores_2(klv_vector,dataset_name,k):
    [N] = np.shape(klv_vector)
    cumul_klv_vector = np.cumsum(klv_vector)
    print(klv_vector)
    plt.figure(figsize=(10, 6)) 
    fig, ax1 = plt.subplots()
    x_axis_list = list(range(N))
    plt.plot(x_axis_list, cumul_klv_vector, color="#3F5D7D")
    #plt.fill_between(x_axis_list,[0]*N,cumul_klv_vector, alpha=0.7)
    plt.plot(x_axis_list, 10*klv_vector, color="r") 
    ax = plt.subplot(111)   
    #ax = plt.axes()        
    ax.yaxis.grid()
    ax.spines["top"].set_visible(False)    
    #ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    #ax.spines["left"].set_visible(False) 
    plt.xlabel("i", fontsize=19)
    plt.ylabel(r'$\sum_{j=1}^{i}\mathrm{\ell}_{j}^{k}$', fontsize=19)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
        
    #plt.title("The k-leverage scores of the matrix "+dataset_name+" (k = "+str(k)+")")
    figfile_name= dataset_name+"_cumul_klv_k_"+str(k)+".pdf"
    plt.savefig(figfile_name)
    plt.show()