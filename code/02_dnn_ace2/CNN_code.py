# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 19:36:27 2025

@author: USER01
"""

import pandas as pd 
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import pickle
from matplotlib.ticker import MaxNLocator

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


# CNN 模型
class CNNRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(64)
        self.fc1 = nn.Linear(16*64, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, 1280]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 数据加载和预处理
df_Desai_5loci_complete=pd.read_csv(DATA_PATH,index_col=0)
df_Desai_5loci_complete = df_Desai_5loci_complete[df_Desai_5loci_complete['log10Kd_ACE2'] != 5]

EMBED='mean_representation'
def list_series_to_tensor(df, col_name):
    return torch.tensor(df[col_name].tolist(), dtype=torch.float32)
df_Desai_5loci_complete[EMBED]=df_Desai_5loci_complete[EMBED].apply(lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')])

features = ['log10Kd_ACE2', 'log10Kd_CB6', 'log10Kd_CoV555', 'log10Kd_REGN10987', 'log10Kd_S309']
seed = 10
df_train = df_Desai_5loci_complete.sample(n=3450, random_state=seed)
df_val = df_Desai_5loci_complete.drop(df_train.index)

train_X = list_series_to_tensor(df_train, EMBED)
train_y = torch.stack([list_series_to_tensor(df_train, f) for f in features], dim=1)

val_X = list_series_to_tensor(df_val, EMBED)
val_y = torch.stack([list_series_to_tensor(df_val, f) for f in features], dim=1)

train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=32)

# CNN 模型实例化
model = CNNRegressor(input_dim=1280, output_dim=5)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 评估模型
model.eval()
with torch.no_grad():
    total_loss = 0
    for batch_X, batch_y in val_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

# 结果可视化
model.eval()
with torch.no_grad():
    val_predictions = model(val_X)
val_predictions_np = val_predictions.numpy()
true_values = val_y.numpy()

feature_names = ["log10Kd_ACE2", "log10Kd_CB6", "log10Kd_CoV555", "log10Kd_REGN10987", "log10Kd_S309"]

plt.figure(figsize=(18, 10))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 3, i+1)
    plt.scatter(true_values[:, i], val_predictions_np[:, i], alpha=0.6)
    plt.plot([min(true_values[:, i]), max(true_values[:, i])], [min(true_values[:, i]), max(true_values[:, i])], color='black', linestyle='--', linewidth=2, label='Perfect fit')
    plt.xlabel('Experimental Values',fontsize=25)
    plt.ylabel('Predicted Values',fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.title(f"Predicted vs True for {feature}",fontsize=18)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax = plt.gca()  # Get current axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.show()


# 创建DataFrame并保存到本地文件夹
for i, feature in enumerate(feature_names):
    # 创建一个 DataFrame
    df = pd.DataFrame({
        'True Values': true_values[:, i],
        'Predicted Values': val_predictions_np[:, i]
    })
    
    # 保存到本地CSV文件，文件名根据特征名称命名
    df.to_csv(os.path.join(OUT_DIR, "CNN_results.csv"), index=False)