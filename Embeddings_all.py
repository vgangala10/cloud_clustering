import numpy as np
import pandas as pd
from model import *
import torch
from torch.utils.data import DataLoader
from Data_loader import *
from config import embedding
from tqdm import tqdm
import os

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")   # Use CPU
print(device)

for file_number in range(100):
    size = embedding['embedding_size']

    # model = TripletLightningModule.load_from_checkpoint(embedding['model_final_path'])
    model = TripletLightningModule.load_from_checkpoint('/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90/lightning_model_50_transform.pt')
    model.to(device)
    model.eval()  # Set the model in evaluation mode (turn off dropout, batch norm, etc.)

    memmaps = np.memmap('/storage/climate-memmap/triplet_data/orig_memmap'+str(file_number)+'.memmap', dtype='float64', mode='r+', shape=(10000, 3, 3, 128, 128))
    data_test_2 = triplet_val(memmaps)
    
    dataloader = DataLoader(data_test_2, batch_size=64, num_workers=0, drop_last=False, shuffle=False)

    memmap_test = np.memmap('/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_60/embeddings'+str(file_number)+'.memmap', dtype='float32', mode='w+', shape=(10000, size))

    all_embeddings = []  # Initialize an empty list to store the embeddings
    print("started_inference")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            patch, _, _ = batch  # Assuming you're interested in the embedding of 'patch'
            patch = patch.to(device)  # Move the patch to GPU if available
            embeddings = model.encode(patch)  # Get the embedding for the 'patch'
            all_embeddings.append(embeddings.cpu().numpy())  # Append the embedding to the list

    all_embeddings = np.vstack(all_embeddings)  # Stack the embeddings into a single numpy array
    memmap_test[:] = all_embeddings[:]  # Write the embeddings to the memmap
    memmap_test.flush()  # Flush changes to disk
    del memmap_test  # Delete the memmap to free up memory
