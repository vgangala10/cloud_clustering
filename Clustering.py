import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import joblib
import os
from config import clustering, embedding

'''
Fits kmeans and gaussian mixture models for 21 K values and stores all the models.
'''

def cluster_t2v(test_embeds, n_clusters):

    if not os.path.exists(clustering['kmeans_path']):
        os.mkdir(clustering['kmeans_path'])
    cluster_model = KMeans(n_clusters=n_clusters, random_state=0).fit(test_embeds)
    joblib.dump(cluster_model, clustering['kmeans_path']+'/kmeans_clustering_model_n-clusters_'+str(n_clusters)+'.joblib')

    # try gaussian mixture model clustering
    if not os.path.exists(clustering['gaussian_path']):
        os.mkdir(clustering['gaussian_path'])
    cluster_model = GaussianMixture(n_components=n_clusters, random_state=0).fit(test_embeds)
    joblib.dump(cluster_model, clustering['gaussian_path']+'/gaussian_mixture_model_n-clusters_'+str(n_clusters)+'.joblib')
    
    return None
test_embeds = np.memmap(embedding['embeddings_memmap_path'], dtype = 'float32', mode = 'r+', shape = (10000*embedding['number_of_files'], embedding['embedding_size']))
for i in range(4,25):
    cluster_t2v(test_embeds, n_clusters = i)