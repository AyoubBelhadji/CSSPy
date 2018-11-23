import sys
#sys.path.append('../')
sys.path.insert(0, '..')


import numpy as np

from sklearn.cluster import KMeans
#from sklearn.metrics import jaccard_similarity_score

from sklearn.metrics.cluster import adjusted_rand_score




def k_means_using_column_subset(clusters_num,X_S):
    #the_matrix = np.dot(X_S,np.linalg.inv(np.sqrt(np.dot(np.transpose(X_S),X_S))))
    the_matrix = X_S
    kmeans = KMeans(init= "k-means++", n_clusters=clusters_num, random_state=0).fit(the_matrix)

    return kmeans.labels_

def evaluate_k_means_using_jaccard(ground_labels,predicted_labels):

    return adjusted_rand_score(ground_labels, predicted_labels)