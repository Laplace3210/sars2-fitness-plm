# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:44:41 2026

@author: USER01
"""


import pandas as pd

# --- Paths -------------------------------------------------------------------
# Run this script from its own directory (code/03_traits/).
import os

PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "../../data/processed")
EXTERNAL_DIR = os.environ.get("EXTERNAL_DIR", "../../data/external")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "../02_dnn_ace2/weights")
# -----------------------------------------------------------------------------


# 你的WT RBD区间（331-531）
wt_rbd = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

def get_rbd_mutations(seq, wt_rbd, start_pos=331):
    rbd_seq = seq[330:531]  # python索引 330~530 共201位
    mutations = []
    for i, (a_wt, a_mut) in enumerate(zip(wt_rbd, rbd_seq)):
        if a_wt != a_mut:
            mutations.append(f"{a_wt}{i+start_pos}{a_mut}")
    return mutations

# 读取数据
df = pd.read_csv(os.path.join(PROCESSED_DIR, "label_mut_seq_merged_updated.csv"))
df['rbd_mut_list'] = df['V2'].apply(lambda seq: get_rbd_mutations(seq, wt_rbd, start_pos=331))
df['rbd_mut_str'] = df['rbd_mut_list'].apply(lambda muts: '-'.join(muts) if muts else "")

# 生成RBD突变序列
from copy import deepcopy

def get_variant_sequence(wt_sequence, mutations):
    wt_sequence_list = list(wt_sequence)
    for mutation in mutations:
        original_aa, position, mutated_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
        # 位置换算
        idx = position - 331
        assert wt_sequence_list[idx] == original_aa, f"Expected {original_aa} at {position}, got {wt_sequence_list[idx]}"
        wt_sequence_list[idx] = mutated_aa
    return ''.join(wt_sequence_list)

df['mutant_rbd_sequence'] = df['rbd_mut_list'].apply(lambda muts: get_variant_sequence(wt_rbd, muts) if muts else wt_rbd)

# 读取 embedding 数据
import pickle
unique_embedding = pickle.load(open(os.path.join(EXTERNAL_DIR, "unique_mutant_sequence_emb_esm1v_650m.pkl"), 'rb'))
unique_embedding = pd.DataFrame(unique_embedding)  # columns: ['sequence', 'mean_representation']

# 对齐字段
unique_embedding = unique_embedding.rename(columns={'sequence': 'mutant_rbd_sequence'})

# merge
merged = pd.merge(df, unique_embedding, on='mutant_rbd_sequence', how='left')


import torch
import torch.nn as nn

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

input_dim = 1280
hidden_dim1 = 64
hidden_dim2 = 32
output_dim = 5
MODEL_PATH = os.path.join(WEIGHTS_DIR, "esm_net_trained_model_5_mut.pt")

model = RegressionNet(input_dim, hidden_dim1, hidden_dim2, output_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# 清洗 embedding
merged = merged.dropna(subset=['mean_representation'])
merged['mean_representation'] = merged['mean_representation'].apply(
    lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')] if isinstance(x, str) else x
)

X_tensor = torch.tensor(list(merged['mean_representation']), dtype=torch.float32)
with torch.no_grad():
    predictions = model(X_tensor).numpy()

merged["pred_log10Kd_ACE2"] = predictions[:, 0]



import re
from escapecalculator import EscapeCalculator

# 已有 merged DataFrame
# merged['rbd_mut_str'] 列为类似 'K417N-E484K-N501Y' 的突变描述

# 1. 提取突变位点号
def extract_positions(mutation_str):
    if pd.isnull(mutation_str) or not mutation_str:
        return []
    return [int(pos) for pos in re.findall(r'\d+', mutation_str)]

# 2. 初始化 EscapeCalculator
calc = EscapeCalculator()

# 3. 计算 binding_retained
def calc_binding_retained(positions):
    if not positions:
        return None
    return float(calc.binding_retained(positions).round(4))

# 4. 批量计算
merged["positions"] = merged["rbd_mut_str"].apply(extract_positions)
merged["binding_retained"] = merged["positions"].apply(calc_binding_retained)


# 5. 保存
merged.to_csv(os.path.join(PROCESSED_DIR, "merged_with_binding_retained_updated.csv"), index=False)


cols_to_export = [
    "lineage", "R", "country", "SDI","spike_binding_score", "spike_escape_score",  
    "spike_entry_score",
    "rbd_mut_str", 
    "positions", 
    "pred_log10Kd_ACE2", 
    "binding_retained"
]
merged[cols_to_export].to_csv(os.path.join(PROCESSED_DIR, "merged_binding_final_updated.csv"), index=False)

