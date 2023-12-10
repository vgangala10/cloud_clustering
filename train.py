import torch
from torch import optim
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from model import *
from Data_loader import *
path_model = '/storage/climate-memmap/models/ResNet34/embedding_75'
if __name__ == '__main__':
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v1")
    torch.set_float32_matmul_precision('medium')
    strategy = DeepSpeedStrategy()
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(path_model+"/tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )
    model = TripletLightningModule(num_blocks=[3, 4, 6, 3], in_channels=3, z_dim=512, lr=0.001, batch_size=16, n_dims=75)
    world_size = torch.cuda.device_count()
    dm = Triplet(
        batch_size=32,
        num_workers=4,
        num_files = 20
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
                        max_epochs=50,
                        profiler = profiler,
                        strategy=DeepSpeedStrategy(logging_batch_size_per_gpu=32),
                        log_every_n_steps = 100,
                        default_root_dir=path_model,
                        callbacks = [checkpoint_callback])  # Set gpus to the number of GPUs
    trainer.fit(model, dm)