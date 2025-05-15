import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class FFCNN_LSTM(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64, output_channels=1, 
                 num_layers=2, kernel_size=3, freq_threshold=60):
        super().__init__()
        self.freq_threshold = freq_threshold
        
        # CNN Encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size, padding='same'),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size, padding='same'),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM Processor
        self.lstm = nn.LSTM(
            input_size=hidden_dim*2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # Distance processing
        self.dist_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # CNN Decoder
        self.final_conv = nn.Sequential(
            nn.Conv1d(hidden_dim * 2 + input_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_channels, kernel_size=1)
        )
    
    def apply_frequency_filter(self, x):
        """Применяет Фурье-фильтр с обнулением частот выше threshold"""
        x_fft = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.size(-1), device=x.device)
        mask = (freqs <= self.freq_threshold).float().to(x.device)
        x_fft_filtered = x_fft * mask.unsqueeze(0).unsqueeze(0)
        return torch.fft.irfft(x_fft_filtered, n=x.size(-1), dim=-1)
    
    def forward(self, x, distances):
        # x shape: [B, 1, T]
        
        # 1. Частотная фильтрация
        x = self.apply_frequency_filter(x)

        # 2. CNN Encoder с сохранением промежуточных результатов
        skip_1 = x  # Сохраняем для skip connection
        x = self.cnn_encoder[0](x)  # Conv1d
        x = self.cnn_encoder[1](x)  # BatchNorm
        x = self.cnn_encoder[2](x)  # ReLU
        x = self.cnn_encoder[3](x)  # MaxPool1d → T//2

        skip_2 = x  # Сохраняем второй уровень
        x = self.cnn_encoder[4](x)  # Вторая Conv1d
        x = self.cnn_encoder[5](x)  # BatchNorm
        x = self.cnn_encoder[6](x)  # ReLU
        x = self.cnn_encoder[7](x)  # MaxPool1d → T//4

        # 3. LSTM processing
        x_enc = x.permute(0, 2, 1)  # [B, T//4, hidden_dim*2]
        lstm_out, _ = self.lstm(x_enc)  # [B, T//4, hidden_dim]

        # 4. Добавляем информацию о расстоянии
        dist_feat = self.dist_proj(distances.unsqueeze(-1))  # [B, hidden_dim]
        lstm_out = lstm_out + dist_feat.unsqueeze(1)  # [B, T//4, hidden_dim]

        # 5. Переворачиваем обратно
        x = lstm_out.permute(0, 2, 1)  # [B, hidden_dim, T//4]

        # 6. Upsample до T//2
        x = F.interpolate(x, size=skip_2.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip_2], dim=1)  # Concatenate по каналам

        # 7. Upsample до T
        x = F.interpolate(x, size=skip_1.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip_1], dim=1)  # Concatenate по каналам

        # 8. Финальная проекция
        output = self.final_conv(x)  # [B, output_channels, T]

        return output