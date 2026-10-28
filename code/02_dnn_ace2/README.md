# Stage 2 — Deep neural networks for ACE2 binding

Four architectures take the 1 × 1,280 ESM-1v embedding as input and are trained against
the combinatorial deep mutational scanning measurements (4,923 variants carrying 1–5
mutations), with a 75% / 25% train–test split.

| Architecture | Script | Output |
|---|---|---|
| **MLP** (used in the paper) | `MLP_code.py` | `MLP_results.csv` |
| CNN | `CNN_code.py` | `CNN_results.csv` |
| LSTM | `LSTM_code.py` | `LSTM_results.csv` |
| RNN | `RNN_code.py` | `RNN_results.csv` |
| all four in one run | `multi_model_comparison.py` | side-by-side training curves |
| apply the trained MLP | `env_5_mut_model.py` | `mut_5_prediction_results.csv`, `unique_5_mut_seq_predictions.csv` |
| comparison figure | `model_comparsion.R` | Figure 3 |

The MLP is `RegressionNet(input_dim=1280, hidden_dim1=64, hidden_dim2=32)` with ReLU
after each hidden layer and no activation on the output, as described in the
Supplementary Methods. It gave the best agreement with experiment
(Pearson correlation coefficient = 0.84) and its predictions are used downstream.

## Trained weights

Included in [`weights/`](weights):

- `esm_net_trained_model_5_mut.pt` — the model used in the paper (1–5 mutations)
- `esm_net_trained_model.pt` — the equivalent model trained on the full 15-locus set

## Running these scripts

Run each script **from this directory** (`code/02_dnn_ace2/`). Paths are resolved
relative to it, and each can be overridden with an environment variable:

| Variable | Default | Contents |
|---|---|---|
| `DESAI_5LOCI` | `../../data/external/df_Desai_5loci.csv` | 4,923 variants with their ESM-1v embeddings (≈137 MB; see [`../01_data_prep/`](../01_data_prep)) |
| `ESM1V_EMBEDDINGS` | `../../data/external/unique_mutant_sequence_emb_esm1v_650m.pkl` | embeddings of the unique circulating RBD sequences |
| `RBD_WITH_TIME` | `../../data/external/rbd_df.csv` | RBD sequences with collection dates, derived from GISAID |
| `WEIGHTS_DIR` | `weights` | trained network weights |
| `MODEL_COMPARISON_OUT` | `../../results/model_comparison` | where results are written |

The three input files under `data/external/` are **not tracked** — two are far above
GitHub's size limits, and `rbd_df.csv` is derived from GISAID sequence records, which
may not be redistributed. Regenerate them as described in
[`../01_data_prep/`](../01_data_prep), or request them from the lead contact; GISAID
records are available to any registered user under the EPI_SET accession listed in
[`../../data/README.md`](../../data/README.md).

> **Not part of this study.** ESM-1b and ESM-2 variants of these comparisons were run
> while exploring the design and are deliberately excluded, because the published
> analysis uses ESM-1v throughout.
