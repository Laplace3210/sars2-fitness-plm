# Stage 3 — Viral traits

Assembles the per-lineage trait table that the fitness models consume: predicted ACE2
binding, antibody binding retained, a cell-entry score and the socio-demographic index.

| Script | Role |
|---|---|
| `label_mut_seq_merged_updated.R` | Maps GISAID lineage records to spike sequences and their RBD mutation sets |
| `merged_binding_final_updated.py` | Applies the trained MLP to each RBD sequence and merges the predicted binding values |
| `combine_anti_retained_and_wang_data.R` | Joins antibody binding retained to the lineage table |
| `all_data_with_SDI_updated.R` | Adds country growth rates and the socio-demographic index |
| `AI_predict_binding.R` | Compares model-predicted and DMS-measured binding across variants of concern (Figure 2E) |

## Antibody binding retained

Computed with the antibody-escape estimator of **Greaney et al. (2022)**:

> Greaney, A.J., Starr, T.N., and Bloom, J.D. (2022). An antibody-escape estimator for
> mutations to the SARS-CoV-2 receptor-binding domain. *Virus Evolution* **8**, veac021.
> <https://doi.org/10.1093/ve/veac021>
>
> Code: **<https://github.com/jbloomlab/SARS2-RBD-escape-calc>**
> Interactive version: <https://jbloomlab.github.io/SARS2-RBD-escape-calc/>

Their `escapecalculator.py` is **not redistributed in this repository**. Download it from
the repository above and place it in this folder before running the stage; it is used
unmodified, with default antibody-set settings.

## Running these scripts

Run each from this directory. Paths are relative and can be overridden:

| Variable | Default |
|---|---|
| `PROCESSED_DIR` | `../../data/processed` |
| `EXTERNAL_DIR` | `../../data/external` |
| `WEIGHTS_DIR` | `../02_dnn_ace2/weights` |

Inputs under `data/external/` are not tracked — they are either above GitHub's size
limits or derived from GISAID records, which may not be redistributed. See
[`../../data/README.md`](../../data/README.md).
