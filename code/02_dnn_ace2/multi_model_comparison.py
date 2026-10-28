# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 19:06:20 2025

@author: USER01
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

# --- Paths -------------------------------------------------------------------
# Run this script from its own directory (code/02_dnn_ace2/), or point the
# environment variables below at your own copies of the files.
import os

DATA_PATH = os.environ.get(
    "DESAI_5LOCI", "../../data/external/df_Desai_5loci.csv")
OUT_DIR = os.environ.get(
    "MODEL_COMPARISON_OUT", "../../results/model_comparison")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "weights")
os.makedirs(OUT_DIR, exist_ok=True)
# -----------------------------------------------------------------------------


# ----------------- 数据加载 ------------------
df = pd.read_csv(DATA_PATH, index_col=0)
df = df[df['log10Kd_ACE2'] != 5]
df['mean_representation'] = df['mean_representation'].apply(lambda x: [float(i) for i in x.strip('[]').split(', ')])

features = ['log10Kd_ACE2', 'log10Kd_CB6', 'log10Kd_CoV555', 'log10Kd_REGN10987', 'log10Kd_S309']
def list_series_to_tensor(df, col): return torch.tensor(df[col].tolist(), dtype=torch.float32)

df_train = df.sample(n=3450, random_state=42)
df_val = df.drop(df_train.index)

train_X = list_series_to_tensor(df_train, 'mean_representation')
val_X = list_series_to_tensor(df_val, 'mean_representation')
train_y = torch.stack([list_series_to_tensor(df_train, f) for f in features], dim=1)
val_y = torch.stack([list_series_to_tensor(df_val, f) for f in features], dim=1)

train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=32)

# ----------------- 模型定义 ------------------
class RegressionNet(nn.Module):
    def __init__(self):  # MLP
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1280, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
    def forward(self, x): return self.model(x)

class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 5)
    def forward(self, x):
        x = x.unsqueeze(-1)  # Reshape input to [batch, seq_len, 1]
        out, _ = self.lstm(x)  # LSTM processing
        return self.fc(out[:, -1, :])  # Take the last output and pass through the fully connected layer

class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(64)
        self.fc1 = nn.Linear(16*64, 64)
        self.fc2 = nn.Linear(64, 5)
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, 1280]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class GRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 5)
    def forward(self, x):
        x = x.unsqueeze(-1)  # Reshape input to [batch, seq_len, 1]
        out, _ = self.gru(x)  # GRU processing
        return self.fc(out[:, -1, :])  # Take the last output and pass through the fully connected layer

# ----------------- 训练评估 ------------------
def train_and_evaluate(model, train_loader, val_loader, epochs=100, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in val_loader:
            pred = model(X)
            preds.append(pred)
            trues.append(y)

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    spearman = [spearmanr(preds[:, i], trues[:, i]).correlation for i in range(5)]
    loss = ((preds - trues) ** 2).mean()
    return loss, spearman

# ----------------- 执行对比 ------------------
models = {
    "MLP": RegressionNet(),
    "LSTM": LSTMRegressor(),
    "CNN": CNNRegressor(),
    "GRU": GRURegressor()  # Using GRU instead of Transformer
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    loss, spearman = train_and_evaluate(model, train_loader, val_loader)
    results[name] = {"MSE Loss": loss, "Avg Spearman": np.mean(spearman), "Spearman (per target)": spearman}

# ----------------- 打印结果 ------------------
for name, res in results.items():
    print(f"\n{name}:\n  MSE Loss = {res['MSE Loss']:.4f}\n  Avg Spearman = {res['Avg Spearman']:.4f}\n  Spearman = {res['Spearman (per target)']}")

# ----------------- 可视化 ------------------
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), [res['Avg Spearman'] for res in results.values()])
plt.title("Average Spearman Correlation per Model")
plt.ylabel("Spearman Correlation")
plt.show()









