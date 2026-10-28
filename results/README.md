# Results

Metric tables produced by the scripts in [`../code/04_fitness_models/`](../code/04_fitness_models).

## Leave-one-country-out analyses

Each folder is one country set. The names match the labels used in the
Supplementary Methods, and the country set of each was verified against the
`HeldOutCountry` column of the files it contains.

| Folder | Countries | Held-out set |
|---|---|---|
| `main_analysis` | **13** | the nine study countries (Bangladesh, Brazil, Cambodia, Germany, India, Senegal, Turkey, UK, USA) + Chile, Poland, Mexico, **China** |
| `sensitivity1` | 12 | as above **without** China |
| `sensitivity2` | 14 | the nine + Chile, Poland, Mexico, **Denmark, Japan** |
| `sensitivity3` | 13 | the nine + Chile, Poland, Mexico, **Nepal** |

Files in each folder:

| File | Contents |
|---|---|
| `classification_metrics_by_country_loco.csv` | One row per held-out country per model |
| `regression_metrics_by_country_loco.csv` | The same for the regression task |
| `classification_metrics_summary_loco.csv` | Mean and 95% CI across held-out countries |
| `regression_metrics_summary_loco.csv` | The same for the regression task |

## External validation

| Folder | Contents |
|---|---|
| `external_validation_9plus1` | Nine-plus-one design, four external countries (Denmark, Japan, Nepal, Mexico) |
| `external_validation_9plus1_3countries` | The same excluding Japan; this is the version reported in the paper |

## Relative-fitness re-analysis

`relative_fitness/` is produced by
[`../code/04_fitness_models/loco_relative_fitness.py`](../code/04_fitness_models/loco_relative_fitness.py)
and is not tracked; run that script to regenerate it.
