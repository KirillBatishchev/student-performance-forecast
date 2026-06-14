import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DKTSequenceDataset(Dataset):
    def __init__(self, sequences, feature_cols, target_col, max_len=50):
        self.sequences = sequences
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        df = self.sequences[idx]
        X = df[self.feature_cols].values
        y = df[self.target_col].values
        if len(X) > self.max_len:
            X = X[-self.max_len:]
            y = y[-self.max_len:]

        seq_len = len(X)

        X_pad = np.zeros((self.max_len, len(self.feature_cols)))
        y_pad = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)

        X_pad[:seq_len] = X
        y_pad[:seq_len] = y
        mask[:seq_len] = 1

        return {
            "X": torch.FloatTensor(X_pad),
            "y": torch.FloatTensor(y_pad),
            "mask": torch.FloatTensor(mask),
            "seq_len": seq_len,
        }


class SimpleDKT(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=64, num_layers=2, n_heads=4, dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(1000, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: [batch, seq_len, features]
        # mask: [batch, seq_len]

        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        src_key_padding_mask = mask == 0
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        logits = self.output(x).squeeze(-1)

        return torch.sigmoid(logits)
