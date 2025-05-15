import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # Encoder
        self.enc1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        # Decoder
        self.dec1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.dec3 = nn.Conv1d(16, 1, kernel_size=3, padding=1)

        # Pooling and Upsampling
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc1(x))
        x2 = self.pool(F.relu(self.enc2(x1)))
        x3 = self.pool(F.relu(self.enc3(x2)))

        # Decoder
        x4 = self.upsample(F.relu(self.dec1(x3)))
        
        # Ensure the size matches for skip connection
        if x4.size(2) != x2.size(2):
            x4 = F.pad(x4, (0, x2.size(2) - x4.size(2)))
        
        x5 = self.upsample(F.relu(self.dec2(x4 + x2)))
        
        if x5.size(2) != x1.size(2):
            x5 = F.pad(x5, (0, x1.size(2) - x5.size(2)))
        
        x6 = self.dec3(x5 + x1)

        return x6