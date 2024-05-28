import numpy as np
from sklearn.cluster import KMeans
import os
import joblib
n_files = list(range(1, 11))
for n_file in n_files:
    filenames = [f'/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90/embeddings_50/embeddings{i}.memmap' for i in range(n_file)]

    # Create a list of memmaps
    memmaps = [np.memmap(filename, dtype='float32', mode='r', shape=(10000, 50)) for filename in filenames]
    concatenated = np.concatenate(memmaps, axis=0)

    def cluster_t2v(test_embeds, n_clusters, n_file):
        os.makedirs(f'/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90/kmeans_50_multi/{n_file}_files', exist_ok=True)
        cluster_model = KMeans(n_clusters=n_clusters, random_state=0).fit(test_embeds)
        joblib.dump(cluster_model, f'/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90/kmeans_50_multi/{n_file}_files'+'/kmeans_clustering_model_n-clusters_'+str(n_clusters)+'.joblib')
    for i in range(4,25):
        cluster_t2v(concatenated, n_clusters = i, n_file = n_file)
