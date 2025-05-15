import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, hidden_dim=32):
        super(SimpleModel, self).__init__()  # Fixed super() call
        # encoder
        self.en1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.en2 = nn.ReLU()
        self.en3 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.en4 = nn.ReLU()
        self.dp1 = nn.Dropout(0.4)

        self.dist_fc = nn.Linear(1, hidden_dim//4)

        # decoder
        self.dc1 = nn.Conv1d(hidden_dim*2 + hidden_dim//4, hidden_dim*2, kernel_size=3, padding=1)
        self.dc2 = nn.ReLU()
        self.dc3 = nn.Conv1d(hidden_dim*2, output_channels, kernel_size=3, padding=1)

    def forward(self, x, distance):
        # Encoder
        x = self.dp1(self.en4(self.en3(self.en2(self.bn1(self.en1(x))))))
        # distance
        
        # Process distance feature
        distance_feat = self.dist_fc(distance.unsqueeze(-1)).unsqueeze(-1)  # [batch, hidden_dim * 2, 1]
        distance_feat = distance_feat.expand(-1, -1, x.shape[-1])  # Match encoder output length
        
        x = torch.cat([x, distance_feat], dim=1)

        #decod
        x = self.dc3(self.dc2(self.dc1(x)))
        return x