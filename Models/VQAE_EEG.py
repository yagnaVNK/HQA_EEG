from nnspt.segmentation.unet import Unet
import torch
import torch.nn as nn
from torchsummary import summary


class VQAE_EEG(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Unet(in_channels=32, out_channels=6, encoder='timm-efficientnet-b1')
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


