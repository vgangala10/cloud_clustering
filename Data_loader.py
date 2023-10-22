import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np

class TripletConcatDataset(Dataset):
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
    
class Triplet(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, num_files = 2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_files = num_files

    def setup(self, stage):
        memmaps = [np.memmap('/storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r+', shape = (10000, 3, 3, 128, 128)) for i in range(self.num_files)]
        data_ALL = ConcatDataset(memmaps)
        self.data = TripletConcatDataset(data_ALL) 

    def train_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )