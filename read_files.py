import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from model import *
import json
from Data_loader import *
from torch.utils.data import DataLoader, Dataset, ConcatDataset

resnet = TripletLightningModule.load_from_checkpoint("/storage/climate-memmap/models/ResNet34/lightning_model_100.pt")
# model.load_state_dict(torch.load("/storage/climate-memmap/lightning_model.pt"))
# model = TripletLightningModule.load_from_checkpoint("/storage/climate-memmap/lightning_logs/version_6/checkpoints/epoch=9-step=31250.ckpt/checkpoint/mp_rank_00_model_states.pt")
resnet.eval()        
dict_all = {}
for j in range(4,15):
    # Get a list of all folders in the specified directory
    model = joblib.load('/storage/climate-memmap/models/ResNet34/cluster_models/kmeans/kmeans_clustering_model_n-clusters_'+str(j)+'.joblib')
    labels = ['Open-cellular_MCC', 'Disorganized_MCC', 'Suppressed_Cu', 'Clustered_cumulus', 'Closed-cellular_MCC', 'Solid_Stratus']
    num_files = [368, 327, 263, 268, 328, 355]
    # List files in each folder
    cluster_all = []
    for i in range(len(labels)):
        cluster_labels = []
        memmap = np.memmap('/storage/climate-memmap/classified_cloud_images_modified/'+labels[i]+'/memmap2.memmap', dtype = 'float64', mode = 'r', shape = (num_files[i], 3, 128, 128))
        for k in range(len(memmap)):
            try:
                patch = np.expand_dims(memmap[k], axis = 0)
                patch = patch.astype(np.float32)
                patch = torch.from_numpy(patch)
                embeddings = resnet.encode(patch)
                embeddings = embeddings.detach().numpy()
                cluster_labels.extend(model.predict(embeddings).tolist())
            except:
                print(k, labels[i])
        cluster_all.append(cluster_labels)  
    dict_all[str(j)] = cluster_all
with open('/storage/climate-memmap/models/ResNet34/cluster_models/kmeans/cluster_classified.json', 'w') as file:
    json.dump(dict_all, file)
    

