# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 21:02:19 2025

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


class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
df_Desai_5loci_complete=pd.read_csv(DATA_PATH,index_col=0)

df_Desai_5loci_complete = df_Desai_5loci_complete[df_Desai_5loci_complete['log10Kd_ACE2'] != 5]

EMBED='mean_representation'
def list_series_to_tensor(df, col_name):
    return torch.tensor(df[col_name].tolist(), dtype=torch.float32)
df_Desai_5loci_complete[EMBED]=df_Desai_5loci_complete[EMBED].apply(lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')])


features = ['log10Kd_ACE2', 'log10Kd_CB6', 'log10Kd_CoV555', 'log10Kd_REGN10987', 'log10Kd_S309']

seed = 10
# sample_size = int(len(df_Desai_5loci_complete) * 0.7)
df_train = df_Desai_5loci_complete.sample(n=3450, random_state=seed)
df_val = df_Desai_5loci_complete.drop(df_train.index)
# Convert dataframes to tensors using the custom function
train_X = list_series_to_tensor(df_train, EMBED)
train_y = torch.stack([
    list_series_to_tensor(df_train, 'log10Kd_ACE2'),
    list_series_to_tensor(df_train, 'log10Kd_CB6'),
    list_series_to_tensor(df_train, 'log10Kd_CoV555'),
    list_series_to_tensor(df_train, 'log10Kd_REGN10987'),
    list_series_to_tensor(df_train, 'log10Kd_S309')
], dim=-1)

val_X = list_series_to_tensor(df_val, EMBED)
val_y = torch.stack([
    list_series_to_tensor(df_val, 'log10Kd_ACE2'),
    list_series_to_tensor(df_val, 'log10Kd_CB6'),
    list_series_to_tensor(df_val, 'log10Kd_CoV555'),
    list_series_to_tensor(df_val, 'log10Kd_REGN10987'),
    list_series_to_tensor(df_val, 'log10Kd_S309')
], dim=-1)
# DataLoader creation
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=32)

import torch.nn as nn
import torch.optim as optim


# Hyperparameters and Model instantiation
input_dim = 1280 
output_dim = len(features)
model = RegressionNet(input_dim, 64, 32, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
model.eval()
with torch.no_grad():
    total_loss = 0
    for batch_X, batch_y in val_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    

from matplotlib.ticker import MaxNLocator
model.eval()
with torch.no_grad():
    val_predictions = model(val_X)
val_predictions_np = val_predictions.numpy()

true_values = val_y.numpy()

# 保存训练后的模型参数
#torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "esm_net_trained_model_5_mut.pt"))

#Plot predicted vs. true values for each Kd
feature_names = ["log10Kd_ACE2", "log10Kd_CB6", "log10Kd_CoV555", "log10Kd_REGN10987", "log10Kd_S309"]

plt.figure(figsize=(18, 10))

for i, feature in enumerate(feature_names):
    plt.subplot(2, 3, i+1)
    plt.scatter(true_values[:, i], val_predictions_np[:, i], alpha=0.6)
    plt.plot([min(true_values[:, i]), max(true_values[:, i])], [min(true_values[:, i]), max(true_values[:, i])],  color='black', linestyle='--', linewidth=2, label='Perfect fit')
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
#plt.savefig('../paper/BP_SI_ML_KD.png', dpi=300)
plt.show()

# 创建DataFrame并保存到本地文件夹
for i, feature in enumerate(feature_names):
    # 创建一个 DataFrame
    df = pd.DataFrame({
        'True Values': true_values[:, i],
        'Predicted Values': val_predictions_np[:, i]
    })
    
    # 保存到本地CSV文件，文件名根据特征名称命名
    df.to_csv(os.path.join(OUT_DIR, "MLP_results.csv"), index=False)