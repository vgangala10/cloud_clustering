# train_model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from Final_code.ResNet import *
from config import embedding
import torchmetrics
import torchvision.models as models

import numpy as np
import torch
from torch.utils.data import Dataset
# from torchvision import datasets, transforms

class MemmapDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, test_split=0.15):
        """
        Args:
            root_dir (string): Directory with all the memmap files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.test_split = test_split
        
        # Assuming each class folder has a single memmap file and folder names are class labels
        self.classes = ['Open-cellular_MCC', 'Disorganized_MCC', 'Suppressed_Cu', 'Clustered_cumulus', 'Closed-cellular_MCC', 'Solid_Stratus']
        num_files = [368, 327, 263, 268, 328, 355]
        # self.classes = ['Open-cellular_MCC', 'Suppressed_Cu', 'Clustered_cumulus', 'Closed-cellular_MCC', 'Solid_Stratus']
        # num_files = [368, 263, 268, 328, 355]
        # self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        rng = np.random.RandomState(42)
        self.samples = []
        for cls in range(len(self.classes)):
            memmap_path = os.path.join(root_dir, self.classes[cls], 'memmap2.memmap')
            # Here, adjust the shape and dtype according to your actual data
            data = np.memmap(memmap_path, dtype='float64', mode='r', shape=(num_files[cls], 3, 128, 128))
            num_samples = data.shape[0]
            # num_samples = num_samples - 20
            indices = np.arange(num_samples)
            rng.shuffle(indices)  # Shuffle indices to ensure random split
            num_samples = num_samples - 20
            
            if split == 'train':
                indices = indices[:int(num_samples * (1 - self.test_split))]
            elif split == 'val':  # 'test'
                indices = indices[int(num_samples * (1-self.test_split)):-20]
            else:
                indices = indices[-20:]
            for i in indices:
                if not np.isnan(data[i]).any():
                    self.samples.append((data[i], self.class_to_idx[self.classes[cls]]))
                    # print(self.class_to_idx[self.classes[cls]])
            
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, label = self.samples[idx]
        patch = patch.squeeze()
        patch = patch.astype(np.float32)
        patch = torch.from_numpy(patch)
        # patch = patch.cuda()
            
        return patch, label
    
# Assuming TripletLightningModule is defined in your model.py
# from model import TripletLightningModule

class Classifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        # Load the pre-trained model
        # self.feature_extractor = TripletLightningModule.load_from_checkpoint('/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90_land/lightning_model_50_transform.pt')
        # # for param in self.feature_extractor.parameters():
        # #     param.requires_grad = False
        # self.layer1 = nn.Linear(50, 50)
        self.feature_extractor = models.vgg16(pretrained=True)
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
        self.layer1 = nn.Linear(1000, 50)
        # Add a classification layer
        self.classifier = nn.Linear(50, num_classes)  # Adjust 512 based on your model's feature size
        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro', task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro', task='multiclass')
        self.f1_score = torchmetrics.F1Score(num_classes=num_classes, average='weighted', task='multiclass')
        self.precision = torchmetrics.Precision(num_classes=num_classes, average='weighted', task='multiclass')
        self.recall = torchmetrics.Recall(num_classes=num_classes, average='weighted', task='multiclass')
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.layer1(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        train_acc = self.train_acc(preds, y)
        train_f1 = self.f1_score(preds, y)
        train_prec = self.precision(preds, y)
        train_recall = self.recall(preds, y)
        self.log('train_loss', loss)
        self.log('train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        val_acc = self.val_acc(preds, y)
        val_f1 = self.f1_score(preds, y)
        val_prec = self.precision(preds, y)
        val_recall = self.recall(preds, y)
        self.log('val_loss', loss)
    def on_train_epoch_end(self):
        # Log training metrics at the end of an epoch
        avg_train_acc = self.train_acc.compute()  # Compute overall accuracy
        self.log('avg_train_acc', avg_train_acc, prog_bar=True)
        self.train_acc.reset()  # Reset metrics for the next epoch

        print(f'\nEpoch {self.current_epoch} - Average Training Accuracy: {avg_train_acc:.4f}')

    def on_validation_epoch_end(self):
        # Log validation metrics at the end of an epoch
        avg_val_acc = self.val_acc.compute()
        avg_f1_score = self.f1_score.compute()
        avg_precision = self.precision.compute()
        avg_recall = self.recall.compute()

        # Log the metrics
        self.log('avg_val_acc', avg_val_acc, prog_bar=True)
        self.log('avg_f1_score', avg_f1_score, prog_bar=True)
        self.log('avg_precision', avg_precision, prog_bar=True)
        self.log('avg_recall', avg_recall, prog_bar=True)

        # Print the metrics
        print(f'\nEpoch {self.current_epoch} - Validation Metrics:\n'
              f'Accuracy: {avg_val_acc:.4f}, '
              f'F1 Score: {avg_f1_score:.4f}, '
              f'Precision: {avg_precision:.4f}, '
              f'Recall: {avg_recall:.4f}')

        # Reset metrics for the next epoch
        self.val_acc.reset()
        self.f1_score.reset()
        self.precision.reset()
        self.recall.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def get_data_loaders(batch_size):
    
    train_dataset = MemmapDataset(root_dir='/storage/climate-memmap/classified_cloud_images_modified', split = 'train')
    val_dataset = MemmapDataset(root_dir='/storage/climate-memmap/classified_cloud_images_modified', split = 'val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader



def main():
    num_classes = 6  
    batch_size = 32
    learning_rate = 1e-3

    train_loader, val_loader = get_data_loaders(batch_size)

    model = Classifier(num_classes=num_classes, learning_rate=learning_rate)
    checkpoint_callback = ModelCheckpoint(monitor='avg_val_acc', save_top_k=1, mode='max')
    logger = TensorBoardLogger("/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90_land/logs_classification_accuracy_D_MCC", name="my_model_logs")
    world_size = torch.cuda.device_count()
    trainer = pl.Trainer(max_epochs=100, accelerator = "gpu", devices = list(range(world_size)), callbacks=[checkpoint_callback], logger=logger)
    # trainer = pl.Trainer(max_epochs=100, accelerator = "cpu", callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()