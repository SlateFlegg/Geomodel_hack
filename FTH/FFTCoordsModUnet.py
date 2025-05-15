import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordsModUnet(nn.Module):
    def __init__(self, input_channels=2, output_channels=1, hidden_dim=8):
        super().__init__()
        # Spatial Feature Encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Encoder Blocks
        self.encoder1 = nn.Sequential(
            nn.Conv1d(2, hidden_dim, kernel_size=7, padding=3),  # Input has two channels now
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ELU(),
            nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=7, padding=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim*4),
            nn.ELU(),
            nn.Conv1d(hidden_dim*4, hidden_dim*4, kernel_size=7, padding=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dim*4 + 8, hidden_dim*4, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim*4),
            nn.ELU()
        )
        
        # Decoder with spatial conditioning
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim*4, hidden_dim*2, kernel_size=2, stride=2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ELU()
        )
        
        self.mid_conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim*4 + hidden_dim, hidden_dim*2, kernel_size=7, padding=3),
            nn.ELU(),
            nn.Dropout(0.1)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim*2, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
        self.mid_conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim*3, hidden_dim*2, kernel_size=7, padding=3),
            nn.ELU(),
            nn.Dropout(0.3)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim*2, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
        self.mid_conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim, input_channels, kernel_size=7, padding=3),
            nn.ELU(),
            nn.Dropout(0.3)
        )
        
        # Final output with spatial context
        self.final_conv = nn.Sequential(
            nn.Conv1d(input_channels + 8, 2, kernel_size=1)
            )

    def forward(self, x, spatial_features):
        # Encode spatial features
        spatial_encoded = self.spatial_encoder(spatial_features)
        spatial_expanded = spatial_encoded.unsqueeze(-1)

        # --- Encoder path ---
        x = x.squeeze(2)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # --- Bottleneck with spatial conditioning ---
        spatial_bottleneck = spatial_expanded.expand(-1, -1, x3.shape[-1])
        x = torch.cat([x3, spatial_bottleneck], dim=1)
        x = self.bottleneck(x)

        # --- Decoder 1 + Skip connection + Spatial Conditioning ---
        x = self.decoder1(x)
        if x.shape[2] != x2.shape[2]:
            diff = x2.shape[2] - x.shape[2]
            x = F.pad(x, (diff // 2, diff - diff // 2))

        spatial_skip1 = spatial_expanded.expand(-1, -1, x2.shape[-1])
        x2_with_spatial = torch.cat([x2, spatial_skip1], dim=1)
        x = torch.cat([x, x2_with_spatial], dim=1)
        x = self.mid_conv1(x)

        # --- Decoder 2 + Skip connection + Spatial Conditioning ---
        x = self.decoder2(x)
        if x.shape[2] != x1.shape[2]:
            diff = x1.shape[2] - x.shape[2]
            x = F.pad(x, (diff // 2, diff - diff // 2))

        spatial_skip2 = spatial_expanded.expand(-1, -1, x1.shape[-1])
        x1_with_spatial = torch.cat([x1, spatial_skip2], dim=1)
        x = torch.cat([x, x1_with_spatial], dim=1)
        x = self.mid_conv2(x)

        # --- Final Decoder Block ---
        x = self.decoder3(x)
        if x.shape[2] != x.shape[2]:
            diff = x.shape[2] - x.shape[2]
            x = F.pad(x, (diff // 2, diff - diff // 2))

        x = self.mid_conv3(x)

        # --- Final output with spatial context ---
        spatial_final = spatial_expanded.expand(-1, -1, x.shape[-1])
        x = torch.cat([x, spatial_final], dim=1)
        x = self.final_conv(x)

        return x