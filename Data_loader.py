import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import json

class TripletConcatDataset(Dataset):
    '''Custom Dataloader with multiple memmaps concatenated using ConcatDataset'''
    def __init__(self, concat_dataset):
        self.concat_dataset = concat_dataset
        self.dataset_lengths = [len(dataset) for dataset in concat_dataset.datasets]
        self.cumulative_lengths = [0] + list(np.cumsum(self.dataset_lengths))

    def __getitem__(self, index):
        # Find the corresponding dataset and index within that dataset
        for i, length in enumerate(self.cumulative_lengths[:-1]):
            if index >= length and index < self.cumulative_lengths[i + 1]:
                dataset_index = i
                dataset_specific_index = index - length

        # Access the corresponding dataset and retrieve the triplet
        specific_dataset = self.concat_dataset.datasets[dataset_index]
        patch, neighbor, distant = specific_dataset[dataset_specific_index]
        patch, neighbor, distant = patch.squeeze(), neighbor.squeeze(), distant.squeeze()
        patch, neighbor, distant = patch.astype(np.float32), neighbor.astype(np.float32), distant.astype(np.float32)
        patch, neighbor, distant = torch.from_numpy(patch), torch.from_numpy(neighbor), torch.from_numpy(distant)
        # patch, neighbor, distant = transform_data(patch), transform_data(neighbor), transform_data(distant)
        # patch, neighbor, distant = patch.to(torch.float32), neighbor.to(torch.float32), distant.to(torch.float32)
        return patch, neighbor, distant

    def __len__(self):
        return sum(self.dataset_lengths)

class TripletConcatDatasetJson(Dataset):
    '''Custom Dataloader with multiple memmaps concatenated using ConcatDataset'''
    def __init__(self, concat_dataset, data_indices):
        self.concat_dataset = concat_dataset
        self.data_indices = data_indices
        self.dataset_lengths = [len(indices) for indices in data_indices]
        self.cumulative_lengths = [0] + list(np.cumsum(self.dataset_lengths))

    def __getitem__(self, index):
        # Find the corresponding dataset and index within that dataset
        for i, length in enumerate(self.cumulative_lengths[:-1]):
            if index >= length and index < self.cumulative_lengths[i + 1]:
                dataset_index = i
                dataset_specific_index = index - length

        # Access the corresponding dataset and retrieve the triplet
        specific_dataset = self.concat_dataset.datasets[dataset_index]
        specific_index = self.data_indices[dataset_index][dataset_specific_index]
        patch, neighbor, distant = specific_dataset[specific_index]
        patch, neighbor, distant = patch.squeeze(), neighbor.squeeze(), distant.squeeze()
        patch, neighbor, distant = patch.astype(np.float32), neighbor.astype(np.float32), distant.astype(np.float32)
        patch, neighbor, distant = torch.from_numpy(patch), torch.from_numpy(neighbor), torch.from_numpy(distant)
        return patch, neighbor, distant
    def __len__(self):
        return sum(self.dataset_lengths)
    
class triplet_val(Dataset):
    '''Data Loading on one memmap or numpy array of size (num_samples, 3, 3, 128, 128)'''
    def __init__(self, data):
        self.val_data = data
        self.length = data.shape[0]
    def __getitem__(self, index):
        patch, neighbor, distant = self.val_data[index]
        patch, neighbor, distant = patch.squeeze(), neighbor.squeeze(), distant.squeeze()
        patch, neighbor, distant = patch.astype(np.float32), neighbor.astype(np.float32), distant.astype(np.float32)
        patch, neighbor, distant = torch.from_numpy(patch), torch.from_numpy(neighbor), torch.from_numpy(distant)
        return patch, neighbor, distant
    def __len__(self):
        return self.length

class Triplet_one(Dataset):
    '''Data Loading for single image memmap for inferencing and similarity search in one array'''
    def __init__(self, data):
        self.data = data
        self.length = data.shape[0]
    def __getitem__(self, index):
        patch = self.data[index]
        patch = patch.squeeze()
        patch = patch.astype(np.float32)
        patch = torch.from_numpy(patch)
        return patch
    def __len__(self):
        return self.length

class Triplet_concat_one(Dataset):
    '''Data loading for single image memmap for infercing and similarity search for multiple memmaps or arrays'''
    def __init__(self, concat_dataset):
        self.concat_dataset = concat_dataset
        self.dataset_lengths = [len(dataset) for dataset in concat_dataset.datasets]
        self.cumulative_lengths = [0] + list(np.cumsum(self.dataset_lengths))
    def __getitem__(self, index):
        for i, length in enumerate(self.cumulative_lengths[:-1]):
            if index >= length and index < self.cumulative_lengths[i + 1]:
                dataset_index = i
                dataset_specific_index = index - length
        specific_dataset = self.concat_dataset.datasets[dataset_index]
        patch,_,_ = specific_dataset[dataset_specific_index]
        patch = patch.squeeze()
        patch = patch.astype(np.float32)
        patch = torch.from_numpy(patch)
        return patch
    def __len__(self):
        return sum(self.dataset_lengths)  

class Triplet(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, num_files = 2, json_file = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_files = num_files
        self.json_file = json_file

    def setup(self, stage):
        memmaps = [np.memmap('/storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128)) for i in range(self.num_files)]
        memmap_val = [np.memmap('/storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128)) for i in range(90, 95)]
        data_val = ConcatDataset(memmap_val)
        data_ALL = ConcatDataset(memmaps)
        if self.json_file!=None:
            data_indices = self.load_data_indices(self.json_file)
            self.data = TripletConcatDatasetJson(data_ALL, data_indices=data_indices[:self.num_files])
            self.val_data = TripletConcatDatasetJson(data_val, data_indices=data_indices[90:95])
        else:
            self.data = TripletConcatDataset(data_ALL) 
            self.val_data = TripletConcatDataset(data_val)
        # memmap_val = np.memmap('/storage/climate-memmap/triplet_data/orig_memmap99.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128))
        # self.val_data = triplet_val(memmap_val)
            
    def load_data_indices(self, json_file):
        if json_file:
            with open(json_file, 'r') as f:
                data_indices = json.load(f)
            return data_indices
        else:
            return None

    def train_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    
class VAEdata(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, num_files = 2, json_file = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_files = num_files
        self.json_file = json_file

    def setup(self, stage):
        memmaps = [np.memmap('/storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128)) for i in range(self.num_files)]
        memmap_val = [np.memmap('/storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128)) for i in range(90, 95)]
        data_val = ConcatDataset(memmap_val)
        data_ALL = ConcatDataset(memmaps)
        if self.json_file!=None:
            data_indices = self.load_data_indices(self.json_file)
            self.data = TripletConcatDatasetJson(data_ALL, data_indices=data_indices[:self.num_files])
            self.val_data = TripletConcatDatasetJson(data_val, data_indices=data_indices[90:95])
        else:
            self.data = Triplet_concat_one(data_ALL) 
            self.val_data = Triplet_concat_one(data_val)
        # memmap_val = np.memmap('/storage/climate-memmap/triplet_data/orig_memmap99.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128))
        # self.val_data = triplet_val(memmap_val)
            
    def load_data_indices(self, json_file):
        if json_file:
            with open(json_file, 'r') as f:
                data_indices = json.load(f)
            return data_indices
        else:
            return None

    def train_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )