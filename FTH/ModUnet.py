import torch
import torch.nn as nn
import torch.nn.functional as F

class ModUnet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, hidden_dim=8):
        super(ModUnet, self).__init__()
        
        # Encoder Blocks
        self.encoder1 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3),  # Changed to explicit padding
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
        
        # Distance processing
        self.dist_fc = nn.Sequential(
            nn.Linear(1, hidden_dim//4),
            nn.ELU()
        )
        
        # Decoder Blocks
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim*4 + hidden_dim//4, hidden_dim*2, kernel_size=2, stride=2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ELU()
        )
        
        self.mid_conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim*4, hidden_dim*2, kernel_size=7, padding=3),
            nn.ELU(),
            nn.Dropout(0.1)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim*2, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
        self.mid_conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=7, padding=3),
            nn.ELU(),
            nn.Dropout(0.3)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, input_channels, kernel_size=2, stride=2),
            nn.BatchNorm1d(input_channels),
            nn.ELU()
        )
        
        self.mid_conv3 = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=7, padding=3),
            nn.ELU(),
            nn.Dropout(0.3)
        )

        
        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv1d(input_channels + 2, output_channels, kernel_size=1)
        )

    def forward(self, x, distance):
        #print(f"\nInput shape: {x.shape}, Distance shape: {distance.shape}")
        
        # Process distance feature
        distance_feat = self.dist_fc(distance.unsqueeze(-1))
        #print(f"Processed distance shape: {distance_feat.shape}")
        
        # Encoder path
        x1 = self.encoder1(x)
        #print(f"After encoder1: {x1.shape}")
        
        x2 = self.encoder2(x1)
        #print(f"After encoder2: {x2.shape}")
        
        x3 = self.encoder3(x2)
        #print(f"After encoder3 (bottleneck): {x3.shape}")
        
        # Prepare distance feature for bottleneck
        distance_bottleneck = distance_feat.unsqueeze(-1).expand(-1, -1, x3.shape[-1])
        #print(f"Distance feature expanded: {distance_bottleneck.shape}")
        
        x = torch.cat([x3, distance_bottleneck], dim=1)
        #print(f"After bottleneck concat: {x.shape}")
        
        # Decoder path
        x = self.decoder1(x)
        #print(f"After decoder1 upsample: {x.shape}")
        
        # Size matching for skip connection
        if x.shape[2] != x2.shape[2]:
            diff = x2.shape[2] - x.shape[2]
            x = F.pad(x, (diff//2, diff - diff//2))
            #print(f"After padding for skip1: {x.shape}")
        
        x = torch.cat([x, x2], dim=1)
        #print(f"After skip1 concat: {x.shape}")
        
        x = self.mid_conv1(x)
        #print(f"After mid_conv1: {x.shape}")
        
        x = self.decoder2(x)
        #print(f"After decoder2 upsample: {x.shape}")
        
        # Size matching for skip connection
        if x.shape[2] != x1.shape[2]:
            diff = x1.shape[2] - x.shape[2]
            x = F.pad(x, (diff//2, diff - diff//2))
            #print(f"After padding for skip2: {x.shape}")
        
        x = torch.cat([x, x1], dim=1)
        #print(f"After skip2 concat: {x.shape}")
        
        x = self.mid_conv2(x)
        #print(f"After mid_conv2: {x.shape}")
        
        x = self.decoder3(x)
        #print(f"After decoder3 upsample: {x.shape}")
        
        x = self.mid_conv3(x)
        #print(f"After mid_conv3: {x.shape}")

        # Final distance conditioning
        distance_final = distance_feat.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        #print(f"Final distance feature shape: {distance_final.shape}")
        
        x = torch.cat([x, distance_final], dim=1)
        #print(f"After final concat: {x.shape}")
        
        x = self.final_conv(x)
        #print(f"Final output shape: {x.shape}")
        
        return x