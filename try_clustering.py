import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
def cluster_t2v(test_embeds, n_clusters, is_plot=False, n_egs=10, savePathVis=None, rgb=True):
    
    # rgb says whether to do plots and cluster_egs using rgb format (True), or input channels (False)
    # clustering is done on embeddings, which must be based on input channels so not effected

    # try ward clustering as per Denby
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', compute_full_tree='auto', linkage='ward')
    cluster_out = cluster_model.fit(test_embeds) # X is array-like, shape (n_samples, n_features) 
    cluster_labels = cluster_out.labels_ # cluster labels directly
    print(cluster_labels[:150])
test_embeds = np.memmap('/storage/climate-memmap/test_embeddings_8100.memmap', dtype = 'float32', mode = 'r+', shape = (20000, 8192))
for i in range(4,5):
    cluster_labels, cluster_egs, cluster_centroids = cluster_t2v(test_embeds,
                                                                            n_clusters = i,
                                                                            is_plot=True, n_egs=10,
                                                                            savePathVis='/storage/climate-memmap/margin_1.0/clusters/t2v_cluster_egs_'+str(i)+'.pdf',rgb=False)