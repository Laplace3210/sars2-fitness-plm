# Stage 4 — Fitness prediction and validation

Nine regression and classification models (Ridge, SVM/SVR, Random Forest, AdaBoost,
XGBoost, LightGBM, CatBoost, a stepwise GAM and a majority-class baseline) are fitted to
four predictors — predicted ACE2 binding, antibody binding retained, cell-entry score and
the socio-demographic index — against the lineage growth rate `R`.

Validation is leave-one-country-out (LOCO): the model is refitted with one country held
out and evaluated on it. Confidence intervals are non-parametric percentile bootstraps
(2,000 resamples).

## Which script produces which result

> The file names carry a historical `sens1` suffix that no longer matches the label used
> in the paper. **This table is authoritative**, and each script states its own analysis
> in a comment at the top of the "Global settings" block.

| Script | Country set | n | Writes to |
|---|---|---|---|
| `country_leave_one_out_full_reg_cls_with_ci_sens1.py` | **the main analysis** — the nine study countries plus Chile, Poland, Mexico and **China** | 13 | `results/main_analysis/` |
| `country_leave_one_out_full_reg_cls_with_ci.py` | the main set **without** China | 12 | `results/sensitivity1/` |
| `country_leave_one_out_full_reg_cls_with_ci_sens2.py` | the main set plus **Denmark and Japan** | 14 | `results/sensitivity2/` |
| `country_leave_one_out_full_reg_cls_with_ci_sens3.py` | the main set with **Nepal** in place of China | 13 | `results/sensitivity3/` |
| `country_external_validation_9plus1.py` | trained on nine countries, tested on **Denmark, Nepal, Mexico** | 9 + 3 | `results/external_validation_9plus1_3countries/` |
| `loco_relative_fitness.py` | LOCO under an absolute (`R > 60`) versus a within-country median definition of high fitness | 13 | `results/relative_fitness/` |

## Running these scripts

Run each from this directory:

```bash
cd code/04_fitness_models
python country_leave_one_out_full_reg_cls_with_ci_sens1.py
```

| Variable | Default | Meaning |
|---|---|---|
| `FITNESS_DATA` | `../../data/processed/merged_binding_final_updated.csv` | The analysis table (tracked; carries every column these models use) |
| `FITNESS_OUT` | `../../results` | Parent of the output folder named in the table above |

`merged_with_binding_retained_updated.csv` is the untrimmed version of the same table and
is not tracked; point `FITNESS_DATA` at it to reproduce the original runs directly.

> **Reproducibility note.** scikit-learn removed the `SAMME.R` AdaBoost algorithm after
> these results were produced, so AdaBoost metrics may differ by up to ~0.18 on recent
> versions. Every other model, including the SVM and CatBoost models reported in the
> paper, reproduces exactly. Pinned versions are in [`../../requirements.txt`](../../requirements.txt).
