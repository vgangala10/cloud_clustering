import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import joblib
import os
def cluster_t2v(n_clusters, is_plot=False, n_egs=10, savePathVis=None, rgb=True):
    cluster_model = joblib.load('/storage/climate-memmap/models/ResNet34/embedding_50/kmeans/kmeans_clustering_model_n-clusters_'+str(n_clusters)+'.joblib')
    cluster_labels = cluster_model.labels_ # cluster labels directly
    cluster_egs=[]
    for i in range(n_clusters):
        egs_i = x_test_np[cluster_labels==i]
        print('no. egs in cluster', i,':', egs_i.shape[0])
        cluster_egs.append(egs_i[:n_egs])
#         cluster_egs_rgb.append(egs_i_rgb[:n_egs])
    cluster_egs = np.array(cluster_egs)
    print(np.shape(cluster_egs))
    cluster_egs = cluster_egs.transpose(0,1,3,4,2)
    print(np.shape(cluster_egs))
    if is_plot:
        # create figure of row
        rows = n_egs
        cols = n_clusters
        fig = plt.figure(figsize=(cols*1.5,rows*1.5))
        for i in range(rows):
            for j in range(cols):
                ax = fig.add_subplot(rows,cols,(i*cols)+j+1)
                ax.imshow(cluster_egs[j,i,:,:,:]) # for general dims
                ax.set_xticks([])
                ax.set_yticks([])
                if i==0: ax.set_title('cluster'+str(j))
        fig.savefig(savePathVis,format='pdf')
        plt.close(fig)
        print('saved at',savePathVis)

    return None
x_test_np = [np.memmap('/storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128)) for i in range(2)]
x_test_np = np.concatenate(x_test_np, axis = 0)
x_test_np = x_test_np[:,:1,...]
x_test_np = np.squeeze(x_test_np)
print(x_test_np.shape)
# os.mkdir('/storage/climate-memmap/models/ResNet34/cluster_models')
for i in range(4,25):
    cluster_t2v(n_clusters = i,
                is_plot=True, n_egs=10,
                savePathVis='/storage/climate-memmap/models/ResNet34/embedding_50/kmeans/t2v_cluster_egs_'+str(i)+'.pdf',rgb=False)