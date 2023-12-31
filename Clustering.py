import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import joblib
import os
def cluster_t2v(test_embeds, n_clusters, is_plot=False, n_egs=10, savePathVis=None, rgb=True):
    
    # rgb says whether to do plots and cluster_egs using rgb format (True), or input channels (False)
    # clustering is done on embeddings, which must be based on input channels so not effected

    # try ward clustering as per Denbychuk et al. 2020
    # cluster_model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', compute_full_tree='auto', linkage='ward')
    # cluster_out = cluster_model.fit(test_embeds) # X is array-like, shape (n_samples, n_features) 
    # cluster_labels = cluster_out.labels_ # cluster labels directly
    # joblib.dump(cluster_model, '/storage/climate-memmap/models/ResNet34/embedding_50/agglomerative_clustering_model_n-clusters_'+str(n_clusters)+'.joblib')

    # try kmeans clustering
    if not os.path.exists('/storage/climate-memmap/models/ResNet34/embedding_50/kmeans'):
        os.mkdir('/storage/climate-memmap/models/ResNet34/embedding_50/kmeans')
    cluster_model = KMeans(n_clusters=n_clusters, random_state=0).fit(test_embeds)
    joblib.dump(cluster_model, '/storage/climate-memmap/models/ResNet34/embedding_50/kmeans/kmeans_clustering_model_n-clusters_'+str(n_clusters)+'.joblib')

    # try gaussian mixture model clustering
    if not os.path.exists('/storage/climate-memmap/models/ResNet34/embedding_50/gaussian_mixture'):
        os.mkdir('/storage/climate-memmap/models/ResNet34/embedding_50/gaussian_mixture')
    cluster_model = GaussianMixture(n_components=n_clusters, random_state=0).fit(test_embeds)
    joblib.dump(cluster_model, '/storage/climate-memmap/models/ResNet34/embedding_50/gaussian_mixture/gaussian_mixture_model_n-clusters_'+str(n_clusters)+'.joblib')
    

    # find n examples of each cluster
#     cluster_egs=[]
#     cluster_egs_rgb=[]
#     cluster_centroids=[] # centroid of each cluster
#     for i in range(n_clusters):
#         egs_i = x_test_np[cluster_labels==i]
# #         egs_i_rgb = x_test_np_rgb[cluster_labels==i]
#         test_embeds_i = test_embeds[cluster_labels==i]
#         cluster_centroids.append(test_embeds_i.mean(axis=0))
#         print('no. egs in cluster', i,':', egs_i.shape[0])
#         cluster_egs.append(egs_i[:n_egs])
# #         cluster_egs_rgb.append(egs_i_rgb[:n_egs])
#     cluster_egs = np.array(cluster_egs)
# #     cluster_egs_rgb = np.array(cluster_egs_rgb)
#     cluster_centroids = np.array(cluster_centroids)
#     print(np.shape(cluster_egs))
#     cluster_egs = cluster_egs.transpose(0,1,3,4,2)
#     print(np.shape(cluster_egs))
#     if is_plot:
#         # create figure of row
#         rows = n_egs
#         cols = n_clusters
#         fig = plt.figure(figsize=(cols*1.5,rows*1.5))
#         for i in range(rows):
#             for j in range(cols):
#                 ax = fig.add_subplot(rows,cols,(i*cols)+j+1)
#                 if rgb:
#                     ax.imshow(cluster_egs_rgb[j,i]) # if rgb
#                 else:
#                     ax.imshow(cluster_egs[j,i,:,:,:]) # for general dims
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 if i==0: ax.set_title('cluster'+str(j))
#         fig.savefig(savePathVis,format='pdf')
#         plt.close(fig)
#         print('saved at',savePathVis)

    return None
# x_test_np = [np.memmap('/storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128)) for i in range(2)]
# x_test_np = np.concatenate(x_test_np, axis = 0)
# x_test_np = x_test_np[:,:1,...]
# x_test_np = np.squeeze(x_test_np)
# print(x_test_np.shape)
# os.mkdir('/storage/climate-memmap/models/ResNet34/embedding_50')
test_embeds = np.memmap('/storage/climate-memmap/models/ResNet34/embedding_50/test_embeddings_50.memmap', dtype = 'float32', mode = 'r+', shape = (20000, 50))
for i in range(4,25):
    cluster_t2v(test_embeds,
                n_clusters = i,
                is_plot=True, n_egs=10,
                savePathVis='/storage/climate-memmap/models/ResNet34/clusters/t2v_cluster_egs_'+str(i)+'.pdf',rgb=False)