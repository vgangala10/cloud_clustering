import numpy as np
import matplotlib.pyplot as plt
import json

with open('/storage/climate-memmap/coordinates_data/kmeans_50_cluster_labels.json', 'r') as file:
    d = json.load(file)

number_of_examples = 10

for n_clusters in range(4,25):
    labels = d[str(n_clusters)]
    memmaps = [np.memmap('/storage/climate-memmap/coordinates_data/data/data_coords_'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (1000, 3, 128, 128)) for i in range(2)]
    data_ALL = np.concatenate(memmaps, axis = 0)
    cluster_egs=[]
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
    fig.savefig('/storage/climate-memmap/coordinates_data/kmeans_50_cluster_egs_'+str(n_clusters)+'.pdf',format='pdf')
    plt.close(fig)
    print('saved at','/storage/climate-memmap/coordinates_data/kmeans_50_cluster_egs_'+str(n_clusters)+'.pdf')