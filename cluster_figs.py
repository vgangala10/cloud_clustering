import numpy as np
import matplotlib.pyplot as plt
import json
import os
from config import clustering

'''
It gives the images of the clusters corresponding to the clusters formed and saves the image in a pdf file.
'''


with open(clustering['kmeans_path']+'/kmeans_labels_95.json', 'r') as file:
    d = json.load(file)

number_of_examples = 10
data_ALL = np.memmap('/storage/climate-memmap/triplet_data/orig_memmap95.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128))
data_ALL = data_ALL[:, 1:2, :, : ,:]
data_ALL = np.squeeze(data_ALL, axis=1)
# data_ALL = np.concatenate(memmaps, axis = 0)
for n_clusters in range(4,25):
    labels = np.array(d[str(n_clusters)])
    cluster_egs=[]
    print(f"shape of data_ALL = {data_ALL.shape}, cluster_group = {n_clusters}")
    print(labels[:100])
    for i in range(n_clusters):
        egs_i = data_ALL[labels==i]
        print('no. egs in cluster', i,':', egs_i.shape[0])
        cluster_egs.append(egs_i[:number_of_examples])
    cluster_egs = np.array(cluster_egs)
    print(np.shape(cluster_egs))
    cluster_egs = cluster_egs.transpose(0,1,3,4,2)
    print(np.shape(cluster_egs))
    # create figure of row
    rows = number_of_examples
    cols = n_clusters
    fig = plt.figure(figsize=(cols*1.5,rows*1.5))
    for i in range(rows):
        for j in range(cols):
            ax = fig.add_subplot(rows,cols,(i*cols)+j+1)
            ax.imshow(cluster_egs[j,i,:,:,:]) # for general dims
            ax.set_xticks([])
            ax.set_yticks([])
            if i==0: ax.set_title('cluster'+str(j))
    if not os.path.exists(clustering['kmeans_path']+'clusters_figs'):
        os.mkdir(clustering['kmeans_path']+'clusters_figs')
    fig.savefig(clustering['kmeans_path']+'clusters_figs/kmeans_cluster_egs_'+str(n_clusters)+'.pdf',format='pdf')
    plt.close(fig)

    print('saved at',clustering['kmeans_path']+'clusters_figs/kmeans_cluster_egs_'+str(n_clusters)+'.pdf')