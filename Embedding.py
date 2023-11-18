from model import *
import torch
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from Data_loader import *


from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

# Path to the checkpoint directory
checkpoint_directory = "/storage/climate-memmap/models/ResNet34/best_model-epoch=43-val_loss=0.00.ckpt"

# Output file path
output_path = "/storage/climate-memmap/models/ResNet34/lightning_model_100.pt"

# Convert the checkpoint directory to a single FP32 state dictionary file
convert_zero_checkpoint_to_fp32_state_dict(checkpoint_directory, output_path)

print("start creating embeddings")

# Initialize the LightningModule and load the trained model from the checkpoint
model = TripletLightningModule.load_from_checkpoint("/storage/climate-memmap/models/ResNet34/lightning_model_100.pt")
# model.load_state_dict(torch.load("/storage/climate-memmap/lightning_model.pt"))
# model = TripletLightningModule.load_from_checkpoint("/storage/climate-memmap/lightning_logs/version_6/checkpoints/epoch=9-step=31250.ckpt/checkpoint/mp_rank_00_model_states.pt")

model.eval()  # Set the model in evaluation mode (turn off dropout, batch norm, etc.)

# Use a DataLoader to load the patch you want to embed (replace with your data loading logic)
# dataloader = DataLoader(...)  # Configure DataLoader for your input data

# memmap_test = np.memmap('/storage/climate-memmap/triplet_data/orig_memmap14.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128))
# memmap_test = memmap_test[:16, ...]
# data_test_2 = triplet_val(memmap_test)

memmaps = [np.memmap('/storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128)) for i in range(2)]
data_ALL = ConcatDataset(memmaps)
data_test_2 = TripletConcatDataset(data_ALL) 

dataloader = DataLoader(data_test_2, batch_size=32, num_workers=0, drop_last=False, shuffle=False)
memmap_test = np.memmap('/storage/climate-memmap/models/ResNet34/test_embeddings_100.memmap', dtype = 'float32', mode = 'w+', shape = (20000, 100))

# Inference loop to get the embedding for a patch
all_embeddings = [] # Initialize an empty tensor to store the embeddings
print("started_inference")
i = 0
with torch.no_grad():
    for batch in dataloader:
        patch, _, _ = batch  # Assuming you're interested in the embedding of 'patch'
        embeddings = model.encode(patch)  # Get the embedding for the 'patch'
        # print(embeddings)
        # 'embeddings' now contains the embedding vector for the input 'patch'
        all_embeddings.append(embeddings.numpy())  # Append the embedding to the list
all_embeddings = np.vstack(all_embeddings)  # Stack the embeddings into a single numpy array
memmap_test[:] = all_embeddings[:]
memmap_test.flush()
# print("memmap_test")
# print(memmap_test)

# Save the embeddings to a file

