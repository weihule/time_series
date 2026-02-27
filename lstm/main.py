import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn


class LoadForecast(nn.Module):
    def __init__(self, input_dim=6, hidden_size=32, num_layers=2, output_seq_len=24):
        super().__init__()
        self.input_dim = input_dim  # 5特征 + 1负荷
        self.output_seq_len = output_seq_len
        
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1  # 工程级优化：加dropout防止过拟合
        )
        
        # Decoder(输入是上一步的负荷预测值)
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        x: [B, 96*5, 6] 拼接后的输入（5特征 + 1负荷）
        return: [B, 24, 1] 预测的未来负荷
        """
        batch_size = x.size(0)
        
        _, (hn, cn) = self.encoder(x)
        
        # 取输入最后一步的负荷值
        decoder_input = x[:, -1:, -1:]      # [B, 1, 1]
        
        outputs = []
        for _ in range(self.output_seq_len):
            dec_out, (hn, cn) = self.decoder(decoder_input, (hn, cn))
            pred_load = self.fc(dec_out)
            

def test():
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, 480, 6).to(device)
    y_true = torch.randn(batch_size, 24, 1).to(device)  # 真实标签
    
    model = LoadForecast(input_dim=6, hidden_size=32, num_layers=2, output_seq_len=24).to(device)
    model.train()
    model(x)
    
    
if __name__ == "__main__":
    test()
    
    
    
