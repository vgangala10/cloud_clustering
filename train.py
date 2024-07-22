import torch
from torch import optim
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from models.autoencoder import *
from Data_loader import *
from config import train

path_model = train['model_directory']

os.makedirs(path_model, exist_ok = True)

if __name__ == '__main__':
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v1")
    torch.set_float32_matmul_precision('medium')
    strategy = DeepSpeedStrategy()
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(path_model+"/tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )
    # model = TripletLightningModule(num_blocks=[3, 4, 6, 3], in_channels=3, z_dim=512, lr=train['learning_rate'], batch_size=train['batch_size'], n_dims=train['embedding_size'])

    model = AELightningModule(lr = train['learning_rate'])

    world_size = torch.cuda.device_count()

    # # Triplet data for Resnet34
    # dm = Triplet(
    #     batch_size= train['batch_size'],
    #     num_workers= train['number_of_workers'],
    #     num_files = train['training_files'],
    #     json_file='./land_data.json'
    # )
    dm = VAEdata(
        batch_size = train['batch_size'],
        num_workers = train['number_of_workers'],
        num_files = train['training_files']
    )

    checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="Validation_epoch_loss",
    mode="min",
    dirpath=path_model,
    filename='best_model-{epoch:02d}-{val_loss:.2f}'
)
    trainer = pl.Trainer(accelerator = "gpu",
                        devices = list(range(world_size)), 
                        max_epochs=train['epochs'],
                        profiler = profiler,
                        strategy=DeepSpeedStrategy(logging_batch_size_per_gpu=train['batch_size']),
                        log_every_n_steps = 100,
                        default_root_dir=path_model,
                        callbacks = [checkpoint_callback])  # Set gpus to the number of GPUs
    trainer.fit(model, dm)