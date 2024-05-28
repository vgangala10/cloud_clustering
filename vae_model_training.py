import torch
import torch.nn as nn
import torch.nn.functional as F
from Data_loader import *
from torch.utils.data import DataLoader, Dataset, ConcatDataset

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.LeakyReLU(inplace=True))
        in_channels = out_channels
        for _ in range(num_convs-1):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ConvBlock2dT(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(ConvBlock2dT, self).__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding = 1))
        layers.append(nn.LeakyReLU(inplace=True))
        in_channels = out_channels
        for _ in range(num_convs-1):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding = 1))
            layers.append(nn.LeakyReLU(inplace=True))
            in_channels = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.blocken1 = ConvBlock(3, 32, 3)
        self.blocken2 = ConvBlock(32, 64, 3)
        self.blocken3 = ConvBlock(64, 128, 3)
        self.blocken4 = ConvBlock(128, 256, 3)
        self.blocken5 = ConvBlock(256, 512, 3)

        self.fcen1 = nn.Linear(512*4*4, latent_dim)
        self.fcen2 = nn.Linear(512*4*4, latent_dim)
        # self.encoder = Encoder(latent_dim)

        self.fc = nn.Linear(latent_dim, 512*4*4)

        self.blockde5 = ConvBlock2dT(512, 256, 3)
        self.blockde4 = ConvBlock2dT(256, 128, 3)
        self.blockde3 = ConvBlock2dT(128, 64, 3)
        self.blockde2 = ConvBlock2dT(64, 32, 3)
        self.blockde1 = ConvBlock2dT(32, 3, 3)
        # self.decoder = Decoder(latent_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def encoder(self, x):
        ## Encoder
        x = self.blocken1(x)
        x = self.blocken2(x)
        x = self.blocken3(x)
        x = self.blocken4(x)
        x = self.blocken5(x)
        x = x.view(x.size(0), -1)
        mean = self.fcen1(x)
        log_var = self.fcen2(x)
        return mean, log_var

    def decoder(self,x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.blockde5(x)
        x = self.blockde4(x)
        x = self.blockde3(x)
        x = self.blockde2(x)
        x = self.blockde1(x)
        return x

    def forward(self, x):
        ## Encoder
        mean, log_var = self.encoder(x)
        ## Reparameterization
        x = self.reparameterize(mean, log_var)
        ## Decoder
        x = self.decoder(x)
        return x, mean, log_var

def loss_function(recon_x, x, mean, log_var):
    BCE = nn.functional.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD

print("build model")

num_files = 50
memmaps = [np.memmap('/storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r', shape = (10000, 3, 3, 128, 128)) for i in range(num_files)]
memmap_val = [np.memmap('/storage/climate-memmap/triplet_data/orig_memmap'+str(i)+'.memmap', dtype = 'float64', mode = 'r', shape = (10000, 3, 3, 128, 128)) for i in range(90, 91)]
data_val = ConcatDataset(memmap_val)
data_ALL = ConcatDataset(memmaps)

train_data_loader = Triplet_concat_one(data_ALL)
val_data_loader = Triplet_concat_one(data_val)
train_loader = DataLoader(
            train_data_loader,
            batch_size=128,
            num_workers=8,
            shuffle=True,
        )
val_loader = DataLoader(
            val_data_loader,
            batch_size=128,
            num_workers=8,
        )
batch_size = 32
total_iterations = len(train_loader.dataset)//batch_size

from tqdm import tqdm
latent_dim = 50
vae = VAE(latent_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-5)

# Example training loop (assuming you have a DataLoader named 'train_loader')
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
vae.to(device)
for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    i = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mean, log_var = vae(batch)
        loss = loss_function(recon_batch, batch, mean, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        i+=1
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in tqdm(val_loader):
            val_batch = val_batch.to(device)
            recon_val_batch, val_mean, val_log_var = vae(val_batch)
            val_loss += loss_function(recon_val_batch, val_batch, val_mean, val_log_var).item()

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}')