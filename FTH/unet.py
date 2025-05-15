import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceBranch(nn.Module):
    def __init__(self, hidden_dim=32):
        super(DistanceBranch, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class FTHUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=32):
        super(FTHUNet, self).__init__()

        # Encoder part (главная ветвь)
        self.enc1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        # Decoder part (главная ветвь)
        self.dec1 = nn.Conv1d(64 + 32, 32, kernel_size=3, padding=1)  # 64 - главная ветвь, 32 - сигнал от DistanceBranch
        self.dec2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.dec3 = nn.Conv1d(16, 1, kernel_size=3, padding=1)

        # Processing distance branch
        self.distance_branch = DistanceBranch()

        # Pooling and upsampling layers
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs, distances):
        x1 = F.relu(self.enc1(inputs))
        x2 = self.pool(F.relu(self.enc2(x1)))
        x3 = self.pool(F.relu(self.enc3(x2)))

        # Process distance using DistanceBranch
        processed_distance = self.distance_branch(distances.view(-1, 1)).unsqueeze(-1)
        expanded_distance = processed_distance.repeat(1, 1, x3.size(-1))  # Extend over time dimension

        # Combine main signal and distance before decoding
        combined = torch.cat([x3, expanded_distance], dim=1)
        
        # Decode
        x4 = self.upsample(F.relu(self.dec1(combined)))
        #Убираем скип-коннекшонс
        if x4.size(2) != x2.size(2):
            x4 = F.pad(x4, (0, x2.size(2) - x4.size(2)))

        x5 = self.upsample(F.relu(self.dec2(x4+x2)))
        if x5.size(2) != x1.size(2):
            x5 = F.pad(x5, (0, x1.size(2) - x5.size(2)))

        x6 = self.dec3(x5+x1)

        return x6