import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from Final_code.ResNet import *
import json
from Data_loader import *
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from config import *
from tqdm import tqdm

resnet = TripletLightningModule.load_from_checkpoint(embedding['model_final_path'])
print(embedding['model_final_path'])
# model.load_state_dict(torch.load("/storage/climate-memmap/lightning_model.pt"))
# model = TripletLightningModule.load_from_checkpoint("/storage/climate-memmap/lightning_logs/version_6/checkpoints/epoch=9-step=31250.ckpt/checkpoint/mp_rank_00_model_states.pt")
resnet.eval()        
dict_all = {}
labels = ['Open-cellular_MCC', 'Disorganized_MCC', 'Suppressed_Cu', 'Clustered_cumulus', 'Closed-cellular_MCC', 'Solid_Stratus']
num_files = [368, 327, 263, 268, 328, 355]
embedds_all = []
for i in range(len(labels)):
    embedds_one = []
    memmap = np.memmap('/storage/climate-memmap/classified_cloud_images_modified/'+labels[i]+'/memmap2.memmap', dtype = 'float64', mode = 'r', shape = (num_files[i], 3, 128, 128))
    memmap = memmap.astype(np.float32)
    data_ = Triplet_one(memmap)
    data_loader = DataLoader(data_, batch_size = 64)
    nan = []
    for _, data in tqdm(enumerate(data_loader, 0)):
        embeddings = resnet.encode(data)
        embeddings = embeddings.detach().numpy()
        if np.isnan(embeddings).sum()>0:
            nan.extend(embeddings)
        else:
            embedds_one.extend(embeddings)
    del memmap
    for j in nan:
        if np.isnan(j).sum()>0:
            print(j, labels[i])
            continue
        else:
            embedds_one.append(j)
    embedds_one = np.array(embedds_one)
    embedd_memmap = np.memmap('/storage/climate-memmap/classified_cloud_images_modified/'+labels[i]+'/embedd_memmap_'+ str(embedding['embedding_size'])+'_.memmap', 
                              dtype = 'float64', 
                              mode = 'w+', 
                              shape = (len(embedds_one), embedding['embedding_size']))
    print(embedd_memmap.shape, labels[i])
    embedd_memmap[:] = embedds_one[:]
    embedd_memmap.flush()
    embedds_all.append(embedds_one)
    del embedd_memmap


d = {}
for j in tqdm(range(4,25)):
    # Get a list of all folders in the specified directory
    cluster_labels = []
    model = joblib.load(clustering['kmeans_path']+'/kmeans_clustering_model_n-clusters_'+str(j)+'.joblib')
    for i in range(len(labels)):
        cluster_labels.append(model.predict(embedds_all[i]).tolist())
    d[str(j)] = cluster_labels
with open(clustering['kmeans_path']+'/embedding_'+str(embedding['embedding_size'])+'_classified_clustering.json', 'w') as file:
    json.dump(d, file)
    

