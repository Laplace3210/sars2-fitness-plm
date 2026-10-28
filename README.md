# Protein language model–derived ACE2 binding improves cross-country SARS-CoV-2 fitness prediction

Code and processed data for:

> Ding, Z., Liang, H., Wang, R., Gu, H., Du, X., Poon, L.L.M., Volz, E., and Yuan, H.-Y.
> *Protein Language Model–Derived ACE2 Binding with Immune Escape Improves the Prediction
> of Cross-Country SARS-CoV-2 Fitness.* (submitted)

The pipeline infers ACE2-binding affinity directly from receptor-binding domain (RBD)
sequences using ESM-1v embeddings and deep neural networks, so that epistasis among
co-occurring mutations is captured rather than summed from single-mutation measurements.
These AI-derived traits are combined with antibody binding retained, a cell-entry score
and the socio-demographic index to predict the epidemiological growth rate of
SARS-CoV-2 lineages across countries.

---

## Repository layout

```
.
├── code/
│   ├── 01_data_prep/         Input subsetting; ESM-1v embeddings are Wang et al.'s
│   ├── 02_dnn_ace2/          CNN / LSTM / MLP / RNN models for ACE2 binding
│   │   └── weights/          Trained MLP weights (.pt)
│   ├── 03_traits/            Antibody binding retained, trait tables, SDI merge
│   ├── 04_fitness_models/    Fitness prediction, LOCO validation, sensitivity
│   └── 05_phylogenetics/     Nextstrain/Augur pipeline and tree plotting
├── figures/                  Scripts that produce the manuscript figures
├── data/
│   ├── processed/            Small analysis-ready tables (tracked)
│   └── external/             Downloaded third-party data (not tracked)
└── results/                  Metric tables produced by the analysis scripts
```

---

## Data availability

**No SARS-CoV-2 sequence data are included in this repository.** Sequences were obtained
from the GISAID EpiCoV database, whose terms of use prohibit redistribution. The exact
set of sequences analysed is identified by a GISAID EPI_SET identifier; see
[`data/README.md`](data/README.md) for that identifier and for instructions on obtaining
every other input.

What *is* included:

| Path | Contents |
|---|---|
| `data/processed/merged_binding_final_updated.csv` | Lineage-level table of the four predictors and the observed growth rate. This is the direct input to every model in `code/04_fitness_models/`. |
| `data/processed/all_data_with_SDI_updated.csv` | Lineage traits merged with the country-level socio-demographic index. |
| `data/processed/spike_bind_escape_entry_updated.csv` | Per-lineage binding, escape and cell-entry scores. |
| `data/processed/*_rates.csv` | Country-specific lineage growth-rate estimates. |
| `results/` | Metric tables underlying the manuscript figures and supplementary tables. |

---

## Reproducing the analysis

```bash
git clone https://github.com/Laplace3210/sars2-fitness-plm.git
cd sars2-fitness-plm
pip install -r requirements.txt
```

The stages are independent; each reads the outputs of the one before it.

**1. ACE2 binding from sequence** (`code/01_data_prep/`, `code/02_dnn_ace2/`)
The mean-pooled 1 × 1,280 ESM-1v representation of each RBD sequence comes from the
published pipeline of [Wang et al. (2024)](https://doi.org/10.1073/pnas.2314518121),
available at <https://github.com/Dianzhuo-Wang/COVID19-Biophysical-Model>; that code is
not duplicated here. `code/01_data_prep/` restricts the combinatorial dataset to variants
carrying 1–5 mutations (4,923 measurements), and `code/02_dnn_ace2/` trains the four
architectures on a 75% / 25% split. The MLP — `RegressionNet(1280, 64, 32)` in
`MLP_code.py` — performed best (Pearson ρ = 0.84) and its predictions are used
downstream; trained weights are in `code/02_dnn_ace2/weights/`.

**2. Viral traits** (`code/03_traits/`)
Antibody binding retained is computed with the antibody-escape estimator of
Greaney et al. Their `escapecalculator.py` is **not duplicated here**; take it from

> Greaney, A.J., Starr, T.N., and Bloom, J.D. (2022). An antibody-escape estimator for
> mutations to the SARS-CoV-2 receptor-binding domain. *Virus Evolution* **8**, veac021.
> <https://doi.org/10.1093/ve/veac021>
>
> Code: **<https://github.com/jbloomlab/SARS2-RBD-escape-calc>**
> Interactive version: <https://jbloomlab.github.io/SARS2-RBD-escape-calc/>

The remaining scripts in `code/03_traits/` assemble the per-lineage trait table and
merge the socio-demographic index.

**3. Fitness prediction and validation** (`code/04_fitness_models/`)

| Script | Country set | Writes to |
|---|---|---|
| `..._with_ci_sens1.py` | **13** — the main analysis, including China | `results/main_analysis/` |
| `..._with_ci.py` | 12 — the main set without China | `results/sensitivity1/` |
| `..._with_ci_sens2.py` | 14 — plus Denmark and Japan | `results/sensitivity2/` |
| `..._with_ci_sens3.py` | 13 — with Nepal in place of China | `results/sensitivity3/` |
| `country_external_validation_9plus1.py` | nine training + 3 external (Denmark, Nepal, Mexico) | `results/external_validation_9plus1_3countries/` |
| `loco_relative_fitness.py` | LOCO under an absolute versus a within-country definition of high fitness | `results/relative_fitness/` |

The file names carry a historical `sens1` suffix that no longer matches the label used in
the paper; the table above is authoritative, and each script states its own analysis in a
comment at the top. See [`code/04_fitness_models/README.md`](code/04_fitness_models/README.md).

**Run each script from its own directory.** All paths are relative and can be overridden
with the environment variables listed in each folder's README (`FITNESS_DATA`,
`FITNESS_OUT`, `PROCESSED_DIR`, `EXTERNAL_DIR`, …).

**4. Phylogenetics** (`code/05_phylogenetics/`)
Per-country Augur pipelines (MAFFT → IQ-TREE 2 → TreeTime) and the R scripts that draw
the time-scaled trees. Needs the GISAID sequences, so it cannot be run from this
repository alone.

**5. Figures** (`figures/`)
`figure5_python.py` redraws the country-level figure entirely in Python; the remaining
scripts are R.

---

## Software

| Tool | Version |
|---|---|
| Python | 3.x |
| PyTorch | 2.7.0 |
| fair-esm (ESM-1v, `esm1v_t33_650M_UR90S_1`) | --- |
| scikit-learn / XGBoost / LightGBM / CatBoost / pyGAM | see `requirements.txt` |
| R | 4.x |
| Clustal Omega | *fill in* |
| Nextstrain Augur | v33.4.0 |
| MAFFT | v7.490 |
| IQ-TREE 2 | v2.0.7 |
| TreeTime | v0.12.1 |

Run `pip freeze > requirements-lock.txt` on the machine used for the published results
and commit that file, so the exact environment is recorded.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite the sources of the input data listed
in [`data/README.md`](data/README.md).

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)). Third-party data retain the
licences of their original sources.
