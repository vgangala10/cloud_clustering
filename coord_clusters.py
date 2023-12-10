from torch.utils.data import DataLoader, Dataset, ConcatDataset
from model import *
from Data_loader import *
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import json
num_files = 2
memmaps = [np.memmap('/storage/climate-memmap/coordinates_data/data/data_coords_'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (1000, 3, 128, 128)) for i in range(num_files)]
data_ALL = ConcatDataset(memmaps)
data = Triplet_concat_one(data_ALL)
data_loader = DataLoader(data, batch_size = 32)
model = TripletLightningModule.load_from_checkpoint('/storage/climate-memmap/models/ResNet34/embedding_50/lightning_model_50.pt')
# model.load_state_dict(torch.load("/storage/climate-memmap/lightning_model.pt"))
# model = TripletLightningModule.load_from_checkpoint("/storage/climate-memmap/lightning_logs/version_6/checkpoints/epoch=9-step=31250.ckpt/checkpoint/mp_rank_00_model_states.pt")
model.eval()
d = {}
for n_clusters in range(4,25):
    labels_cluster = []
    cluster_model = joblib.load('/storage/climate-memmap/models/ResNet34/embedding_50/kmeans/kmeans_clustering_model_n-clusters_'+str(n_clusters)+'.joblib')
    for _, data in enumerate(data_loader, 0):
        data_model = model.encode(data)
        data_model = data_model.detach().numpy()
        labels_batch = cluster_model.predict(data_model)
        labels_cluster.extend(labels_batch.tolist())
    d[n_clusters] = labels_cluster
with open('/storage/climate-memmap/coordinates_data/kmeans_50_cluster_labels.json', 'w') as file:
    json.dump(d, file)

    
    
    

