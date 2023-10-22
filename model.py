import torch
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn

class ResidualBlock(pl.LightningModule):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class TripletLightningModule(pl.LightningModule):
    def __init__(self, num_blocks, in_channels=3, z_dim=512, lr=0.001, batch_size=8):
        super(TripletLightningModule, self).__init__()
        self.save_hyperparameters()
        self.losses = []
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.in_planes = 64
        self.losses = 0
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=1)
        self.layer5 = self._make_layer(self.z_dim, num_blocks[4],
            stride=2)

    def _make_layer(self, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def encode(self, x):
        # a forward pass
        # x = x.cuda()
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.avg_pool2d(x, 4)
        z = x.reshape(x.size(0), -1)
        return z
    def forward(self, x):
        return self.encode(x)

    def triplet_loss(self, z_p, z_n, z_d, margin=0.1, l2=0):
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + margin)
        l_n = torch.mean(l_n) # does this average over batch?
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd

    def loss(self, patch, neighbor, distant, margin=0.1, l2=0):
        z_p, z_n, z_d = self.forward(patch), self.forward(neighbor), self.forward(distant)
        return self.triplet_loss(z_p, z_n, z_d, margin=margin, l2=l2)

    def training_step(self, batch, batch_idx):
        patch, neighbor, distant = batch
        patch, neighbor, distant = patch.cuda(), neighbor.cuda(), distant.cuda()
        loss, l_n, l_d, l_nd = self.loss(patch, neighbor, distant)
        self.losses += loss.data.item()
        self.log("training_loss_cum", self.losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist = True)
        self.log("training_loss_step", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist = True)
        return loss
    
    def on_train_epoch_end(self):
        self.log_dict(
            {
                "epoch_loss": self.losses
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger = True,
            sync_dist = True
        )
        self.losses = 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer