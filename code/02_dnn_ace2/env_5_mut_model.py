# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:57:27 2025

@author: USER01
"""

import pandas as pd
import torch
import torch.nn as nn
import pickle

# --- Paths -------------------------------------------------------------------
# Run this script from its own directory (code/02_dnn_ace2/), or point the
# environment variables below at your own copies of the files.
import os

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "weights")
DATA_PATH = os.environ.get(
    "DESAI_5LOCI", "../../data/external/df_Desai_5loci.csv")
EMB_PKL = os.environ.get(
    "ESM1V_EMBEDDINGS",
    "../../data/external/unique_mutant_sequence_emb_esm1v_650m.pkl")
RBD_CSV = os.environ.get("RBD_WITH_TIME", "../../data/external/rbd_df.csv")
OUT_DIR = os.environ.get(
    "MODEL_COMPARISON_OUT", "../../results/model_comparison")
os.makedirs(OUT_DIR, exist_ok=True)
# -----------------------------------------------------------------------------

# ------------------------------
# 模型定义（原先的 RegressionNet）
# ------------------------------
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

# ------------------------------
# 参数设置
# ------------------------------
input_dim = 1280  # ESM1v embedding size
hidden_dim1 = 64
hidden_dim2 = 32
output_dim = 5  # 对应5个log10Kd值

# 模型路径和数据路径（你可以替换成自己的路径）
MODEL_PATH = os.path.join(WEIGHTS_DIR, "esm_net_trained_model_5_mut.pt")
CSV_PATH = DATA_PATH
OUTPUT_CSV = os.path.join(OUT_DIR, "mut_5_prediction_results.csv")

# ------------------------------
# 加载模型
# ------------------------------
model = RegressionNet(input_dim, hidden_dim1, hidden_dim2, output_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# ------------------------------
# 读取数据并预处理 mean_representation
# ------------------------------
df = pd.read_csv(CSV_PATH)

# 清洗并转为浮点向量
df["mean_representation"] = df["mean_representation"].apply(
    lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')]
)

# 构造模型输入
X_tensor = torch.tensor(df["mean_representation"].tolist(), dtype=torch.float32)

# ------------------------------
# 模型预测
# ------------------------------
with torch.no_grad():
    predictions = model(X_tensor).numpy()

# 添加预测结果列
df["pred_log10Kd_ACE2"] = predictions[:, 0]
df["pred_log10Kd_CB6"] = predictions[:, 1]
df["pred_log10Kd_CoV555"] = predictions[:, 2]
df["pred_log10Kd_REGN10987"] = predictions[:, 3]
df["pred_log10Kd_S309"] = predictions[:, 4]

# 保存结果
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Done! Prediction results saved to: {OUTPUT_CSV}")


with open(EMB_PKL, 'rb') as file:
    unique_embedding= pickle.load(file)
unique_embedding = pd.DataFrame(unique_embedding)
rbd_withtime = pd.read_csv(RBD_CSV)


import pandas as pd
# Rename 'seq' column to 'mutant_sequence'
rbd_withtime.rename(columns={'seq': 'mutant_sequence'}, inplace=True)

# Convert the 'date' column to datetime
rbd_withtime['date'] = pd.to_datetime(rbd_withtime['date'], format='%Y-%m-%d', errors='coerce')

earliest_dates_df = rbd_withtime.groupby('mutant_sequence').agg(
    earliest_date=('date', 'min'),          
    q05=('date', lambda x: x.quantile(0.05).normalize() if pd.notnull(x.quantile(0.05)) else pd.NaT)
).reset_index()

#rename sequence to mutant_sequence
unique_embedding = unique_embedding.rename(columns={'sequence':'mutant_sequence'})
unique_embedding = unique_embedding.merge(earliest_dates_df, on='mutant_sequence', how='left')

wt = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

unique_embedding['diff'] = unique_embedding['mutant_sequence'].apply(
    lambda x: [(wt[i], i+331, x[i]) for i in range(len(x)) if x[i] != wt[i]])



unique_X_tensor = torch.tensor(unique_embedding["mean_representation"].tolist(), dtype=torch.float32)


# 模型预测
# ------------------------------
with torch.no_grad():
    unique_seq_predictions = model(unique_X_tensor).numpy()


# 添加预测结果列
unique_embedding["pred_log10Kd_ACE2"] = unique_seq_predictions[:, 0]
unique_embedding["pred_log10Kd_CB6"] = unique_seq_predictions[:, 1]
unique_embedding["pred_log10Kd_CoV555"] = unique_seq_predictions[:, 2]
unique_embedding["pred_log10Kd_REGN10987"] = unique_seq_predictions[:, 3]
unique_embedding["pred_log10Kd_S309"] = unique_seq_predictions[:, 4]



# 保存结果
unique_embedding.to_csv(os.path.join(OUT_DIR, "unique_5_mut_seq_predictions.csv"), index=False)

def get_variant_sequence(wt_sequence, mutations):
    wt_sequence_list = list(wt_sequence)
    
    # Apply each mutation to the wild type sequence
    for mutation in mutations:
        # Parse the mutation string (e.g., 'K417N')
        original_aa, position, mutated_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
        
        # Check if the original amino acid matches the wild type sequence at the specified position
        if wt_sequence_list[position - 331] != original_aa:
            raise ValueError(f"Expected {original_aa} at position {position}, but found {wt_sequence_list[position - 331]} in the wild type sequence.")
        # Perform the mutation
        wt_sequence_list[position - 331] = mutated_aa
    
    # Convert the list back to a string to get the final variant sequence
    variant_sequence = ''.join(wt_sequence_list)
    return variant_sequence

alpha_sequence = get_variant_sequence(wt, ['N501Y'])
delta_sequence = get_variant_sequence(wt, [ 'T478K', 'E484K'])
beta_sequence = get_variant_sequence(wt,['K417N', 'E484K', 'N501Y'])
gamma_sequence = get_variant_sequence(wt,['K417T', 'E484K', 'N501Y'])
omicronBA1_sequence = get_variant_sequence(wt,['G339D','S371L','S373P','S375F','K417N','N440K','S477N','T478K','E484A','Q493R','G496S','Q498R','N501Y','Y505H'])
omicronBA2_sequence = get_variant_sequence(wt,['G339D','S371F','S373P','S375F','T376A','D405N','R408S','K417N','N440K','S477N','T478K','E484A','Q493R','Q498R','N501Y','Y505H'])
omicronBA3_sequence = get_variant_sequence(wt,['G339D','S371F','S373P','S375F','D405N','K417N','N440K','G446S','S477N','T478K','E484A','Q493R','Q498R','N501Y','Y505H'])
omicronBA4_sequence = get_variant_sequence(wt,['G339D','S371F','S373P','S375F','T376A','D405N','R408S','K417N','N440K','L452R','S477N','T478K','E484A','F486V','Q498R','N501Y','Y505H'])
XBB_sequence = get_variant_sequence(wt,['G339H', 'R346T', 'L368I', 'S371F', 'S373P', 'S375F', 'T376A', 'D405N','R408S', 'K417N', 'N440K', 'V445P', 'G446S', 'N460K', 'S477N', 'T478K', 'E484A', 'F486S', 'F490S', 'Q498R','N501Y','Y505H'] )
variant_sequences = [delta_sequence, omicronBA2_sequence, omicronBA3_sequence, omicronBA4_sequence]

#["pred_log10Kd_ACE2"]
unique_embedding[unique_embedding['mutant_sequence']==alpha_sequence]["pred_log10Kd_ACE2"]
unique_embedding[unique_embedding['mutant_sequence']==gamma_sequence]["pred_log10Kd_ACE2"]
unique_embedding[unique_embedding['mutant_sequence']==beta_sequence]["pred_log10Kd_ACE2"]
unique_embedding[unique_embedding['mutant_sequence']==delta_sequence]["pred_log10Kd_ACE2"]
unique_embedding[unique_embedding['mutant_sequence']==omicronBA1_sequence]["pred_log10Kd_ACE2"]
unique_embedding[unique_embedding['mutant_sequence']==omicronBA2_sequence]["pred_log10Kd_ACE2"]
unique_embedding[unique_embedding['mutant_sequence']==omicronBA3_sequence]["pred_log10Kd_ACE2"]
unique_embedding[unique_embedding['mutant_sequence']==omicronBA4_sequence]["pred_log10Kd_ACE2"]






















