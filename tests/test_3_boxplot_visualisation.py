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
dataset_name = "leukemia"

exp_number = 50
k = 10

savefile_name = "boxplots/"+dataset_name+"_boosting_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".txt"
boosting_error_fro_aggregated_list_load = np.loadtxt(savefile_name)
boosting_error_fro_aggregated_list_load_to_list = []
for i in list(range(5)):
    boosting_error_fro_aggregated_list_load_to_list.append(boosting_error_fro_aggregated_list_load[i])


#savefile_name_2 = "20k/"+dataset_name+"_boosting_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".txt"
#boosting_error_fro_aggregated_list_load_2 = np.loadtxt(savefile_name)
#boosting_error_fro_aggregated_list_load_to_list.append(boosting_error_fro_aggregated_list_load_2[4])


plt.figure(figsize=(10, 8)) 
plt.xticks(fontsize=22)
plt.yticks(fontsize=16)
ax = plt.subplot(111) 
box_2 = plt.boxplot(boosting_error_fro_aggregated_list_load_to_list, showfliers=False)
plt.setp(box_2['medians'], color='red', linewidth=3)
#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP], showfliers=False)

plt.ylabel(r'$\mathrm{\|\| X- \pi_{S}^{Fr} X \|\| _{Fr}}$', fontsize=18)
#plt.legend(bbox_to_anchor=(1.04,1), loc="bottomleft", inset=.02, ["Volume Sampling","Projection DPP"])
#plt.legend("bottomleft", inset=.02, c("Volume Sampling","Projection DPP"), fill=topo.colors(3), horiz=TRUE, cex=0.8)
#plt.gca().xaxis.set_ticklabels(["Uniform Sampling","LVScores Sampling","Volume Sampling","Projection DPP"])

#plt.gca().xaxis.set_ticklabels(["Uniform Sampling","Volume Sampling","Optimized Projection DPP","Projection DPP","Largest lvs","Pivoted QR","derandomized_volume_sampling","Double Phase"])
plt.xticks(rotation=45)
plt.gca().xaxis.set_ticklabels(["Volume S.","Projection DPP","Largest lvs","Pivoted QR","Double Phase"])



figfile_name= dataset_name+"_boosting_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".pdf"
plt.savefig(figfile_name)

plt.show()










savefile_name = "boxplots/"+dataset_name+"_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".txt"
error_fro_aggregated_list_load = np.loadtxt(savefile_name)
error_fro_aggregated_list_load_to_list = []
for i in list(range(5)):
    error_fro_aggregated_list_load_to_list.append(error_fro_aggregated_list_load[i])


#savefile_name_2 = "20k/"+dataset_name+"_boosting_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".txt"
#boosting_error_fro_aggregated_list_load_2 = np.loadtxt(savefile_name)
#boosting_error_fro_aggregated_list_load_to_list.append(boosting_error_fro_aggregated_list_load_2[4])


plt.figure(figsize=(10, 8)) 
plt.xticks(fontsize=22)
plt.yticks(fontsize=16)
ax = plt.subplot(111) 
box_2 = plt.boxplot(error_fro_aggregated_list_load_to_list, showfliers=False)
plt.setp(box_2['medians'], color='red', linewidth=3)
#plt.boxplot([error_fro_list_uniform_sampling,error_fro_list_volume_sampling,error_fro_list_optimized_projection_DPP], showfliers=False)

plt.ylabel(r'$\mathrm{\|\| X- \pi_{S}^{Fr} X \|\| _{Fr}}$', fontsize=18)
#plt.xlabel()
#plt.legend(bbox_to_anchor=(1.04,1), loc="bottomleft", inset=.02, ["Volume Sampling","Projection DPP"])
#plt.legend("bottomleft", inset=.02, c("Volume Sampling","Projection DPP"), fill=topo.colors(3), horiz=TRUE, cex=0.8)
#plt.gca().xaxis.set_ticklabels(["Uniform Sampling","LVScores Sampling","Volume Sampling","Projection DPP"])

#plt.gca().xaxis.set_ticklabels(["Uniform Sampling","Volume Sampling","Optimized Projection DPP","Projection DPP","Largest lvs","Pivoted QR","derandomized_volume_sampling","Double Phase"])
plt.xticks(rotation=45)
plt.gca().xaxis.set_ticklabels(["Volume S.","Projection DPP","Largest lvs","Pivoted QR","Double Phase"])



figfile_name= dataset_name+"_allalgos_"+str(exp_number)+"samples_k_"+str(k)+".pdf"
plt.savefig(figfile_name)

plt.show()