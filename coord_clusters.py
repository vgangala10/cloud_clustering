from torch.utils.data import DataLoader, Dataset, ConcatDataset
from Final_code.ResNet import *
from Data_loader import *
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from config import clustering, embedding
from tqdm import tqdm
'''
Store embeddings of test data
'''
# num_files = 2
memmap = np.memmap('/storage/climate-memmap/triplet_data/orig_memmap95.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128))
# memmap_test = np.memmap('/storage/climate-memmap/models/ResNet34/embedding_50/test_embeddings_50_coords.memmap', dtype = 'float32', mode = 'r+', shape = (20000, 50))
# memmap_test = ConcatDataset(memmap_test)
data = triplet_val(memmap)
data_loader = DataLoader(data, batch_size = 64)
model = TripletLightningModule.load_from_checkpoint(embedding['model_final_path'])
# model.load_state_dict(torch.load("/storage/climate-memmap/lightning_model.pt"))
# model = TripletLightningModule.load_from_checkpoint("/storage/climate-memmap/lightning_logs/version_6/checkpoints/epoch=9-step=31250.ckpt/checkpoint/mp_rank_00_model_states.pt")
model.eval()
data_embedd = []
for _, data in tqdm(enumerate(data_loader, 0)):
    data_model = model.encode(data[0])
    data_model = data_model.detach().numpy()
    data_embedd.extend(data_model.tolist())
data_embedd = np.array(data_embedd)
mem_embedd = np.memmap(embedding['path']+'test_embedds_10000.memmap',
                       dtype = 'float32',
                       mode = 'w+',
                       shape = (10000, embedding['embedding_size']))
mem_embedd[:] = data_embedd[:]
mem_embedd.flush()


d = {}
for n_clusters in range(4,25):
    # labels_cluster = []
    model_path = clustering['kmeans_path']+'/kmeans_clustering_model_n-clusters_'+str(n_clusters)+'.joblib'
    cluster_model = joblib.load(model_path)
    for _, data in enumerate(data_loader, 0):
        data_model = model.encode(data[0])
        data_model = data_model.detach().numpy()
        labels_batch = cluster_model.predict(data_model)
        labels_cluster.extend(labels_batch.tolist())
    labels_cluster = cluster_model.predict(data)
    d[n_clusters] = labels_cluster
json_path = clustering['kmeans_path']+'/kmeans_labels_95.json'
with open(json_path, 'w') as file:
    json.dump(d, file)