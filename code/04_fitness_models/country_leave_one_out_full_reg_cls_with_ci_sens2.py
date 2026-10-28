# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:00:59 2026

@author: USER01
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:59:45 2026

@author: USER01
"""

# -*- coding: utf-8 -*-
"""
Country leave-one-country-out (LOCO) modeling script

Use only these 9 countries:
    Bangladesh, Brazil, Cambodia, Germany, India, Senegal, Turkey, UK, USA

Validation strategy:
    In each fold, leave out 1 whole country as validation set,
    and train on the remaining 8 countries.

Tasks:
1) Regression on R
2) Binary classification on fitness_class = (R > CLASS_CUTOFF)

Outputs:
- regression_metrics_by_country_loco.csv
- regression_metrics_summary_loco.csv
- regression_metrics_overall_loco.csv
- regression_predictions_loco.csv
- regression_gam_best_splines_loco.csv
- classification_metrics_by_country_loco.csv
- classification_metrics_summary_loco.csv
- classification_metrics_overall_loco.csv
- classification_predictions_loco.csv
- classification_gam_best_splines_loco.csv

Notes:
- "by_country"   = one metric row per held-out country per model
- "summary"      = mean/std/95% CI across held-out countries (fold-level summary)
- "overall"      = pooled point estimate using all held-out predictions together
                    (sample-level pooled estimate, not the mean of country metrics)

Recommended packages:
    pip install pygam xgboost lightgbm catboost
"""

import os
import warnings
import operator
from functools import reduce

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t
from pygam import LinearGAM, LogisticGAM, s, te

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
)
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBRegressor, XGBClassifier

# Optional dependencies
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


# =========================
# 0. Global settings
# =========================
# This script runs the Sensitivity 2 - fourteen countries (main set plus Denmark and Japan).
# Its output folder is results/sensitivity2/.

DATA_PATH = os.environ.get(
    "FITNESS_DATA", "../../data/processed/merged_binding_final_updated.csv")
# The table above is the trimmed analysis file shipped with this repository. It
# carries every column these models use. Point FITNESS_DATA at
# merged_with_binding_retained_updated.csv to run on the full table instead.
SEED = 42
APPLY_R_FILTER = True
R_MIN = 10
R_MAX = 200
CLASS_CUTOFF = 60

FEATURES = [
    'pred_log10Kd_ACE2',
    'binding_retained',
    'spike_entry_score',
    'SDI'
]
TARGET = 'R'
COUNTRY_COL = 'country'

MODEL_COUNTRIES = [
    'Bangladesh', 'Brazil', 'Cambodia', 'Germany',
    'India', 'Senegal', 'Turkey', 'UK', 'USA','Chile', 'Poland', 'Mexico', #'China', 
    'Denmark', 'Japan'#,
        # 'Nepal'
]

OUTPUT_DIR_NAME = "sensitivity2"


# =========================
# 1. Utility helpers
# =========================
def get_output_dir():
    base_dir = os.environ.get("FITNESS_OUT", "../../results")
    output_dir = os.path.join(base_dir, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def prepare_data(df):
    df = df.copy()

    required_cols = FEATURES + [TARGET, COUNTRY_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].replace([np.inf, -np.inf], np.nan)

    if APPLY_R_FILTER:
        df = df[(df[TARGET] >= R_MIN) & (df[TARGET] <= R_MAX)].copy()

    df = df[df[COUNTRY_COL].isin(MODEL_COUNTRIES)].copy()
    df = df.dropna(subset=FEATURES + [TARGET, COUNTRY_COL]).copy()
    df['fitness_class'] = (df[TARGET] > CLASS_CUTOFF).astype(int)
    return df


def safe_pearsonr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    return pearsonr(y_true, y_pred)[0]


def get_regression_metrics(y_true, y_pred):
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
        'Pearson_r': float(safe_pearsonr(y_true, y_pred)) if not np.isnan(safe_pearsonr(y_true, y_pred)) else np.nan,
    }


def get_classification_metrics(y_true, y_pred):
    return {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def summarize_fold_metrics(df, metric_cols, task_name):
    if df.empty:
        return pd.DataFrame()

    rows = []
    for model_name, g in df.groupby('Model'):
        row = {'Task': task_name, 'Model': model_name, 'n_folds': int(len(g))}
        for metric in metric_cols:
            vals = pd.to_numeric(g[metric], errors='coerce').dropna().values
            n = len(vals)
            mean_v = np.nanmean(vals) if n > 0 else np.nan
            std_v = np.nanstd(vals, ddof=1) if n > 1 else np.nan
            se_v = std_v / np.sqrt(n) if n > 1 else np.nan
            if n > 1:
                tcrit = t.ppf(0.975, df=n - 1)
                ci_low = mean_v - tcrit * se_v
                ci_high = mean_v + tcrit * se_v
            else:
                ci_low, ci_high = np.nan, np.nan

            row[f'{metric}_mean'] = mean_v
            row[f'{metric}_std'] = std_v
            row[f'{metric}_ci95_low'] = ci_low
            row[f'{metric}_ci95_high'] = ci_high
        rows.append(row)

    return pd.DataFrame(rows).sort_values('Model').reset_index(drop=True)


# =========================
# 2. GAM helpers
# =========================
def choose_best_terms(X_train, y_train, feature_names, gam_class, k_range=range(4, 9)):
    main_effects = []
    interactions = []
    k_dict = {}

    sdi_idx = feature_names.index('SDI')

    # Main effects
    for idx, name in enumerate(feature_names):
        best_k = None
        best_aic = np.inf
        for k in k_range:
            try:
                gam = gam_class(s(idx, n_splines=k)).fit(X_train, y_train)
                aic = gam.statistics_['AIC']
                if aic < best_aic:
                    best_aic = aic
                    best_k = k
            except Exception:
                continue

        if best_k is not None:
            main_effects.append(s(idx, n_splines=best_k))
            k_dict[name] = best_k

    # Interaction terms: each predictor x SDI
    for idx, name in enumerate(feature_names):
        if name == 'SDI':
            continue
        best_k = None
        best_aic = np.inf
        for k in k_range:
            try:
                gam = gam_class(te(idx, sdi_idx, n_splines=k)).fit(X_train, y_train)
                aic = gam.statistics_['AIC']
                if aic < best_aic:
                    best_aic = aic
                    best_k = k
            except Exception:
                continue

        if best_k is not None:
            interactions.append(te(idx, sdi_idx, n_splines=best_k))
            k_dict[f'{name}:SDI'] = best_k

    return main_effects + interactions, k_dict


def stepwise_gam_fit(X, y, terms, gam_class, threshold=0.05):
    if len(terms) == 0:
        return None, []

    terms_work = terms.copy()

    while len(terms_work) > 0:
        try:
            gam = gam_class(reduce(operator.add, terms_work)).fit(X, y)
        except Exception as e:
            print(f"[Stepwise-GAM] fit failed during stepwise selection: {e}")
            return None, []

        pvals = np.array(gam.statistics_['p_values'][:-1])
        if len(pvals) == 0 or np.all(pvals < threshold):
            break

        idx_to_remove = int(np.argmax(pvals))
        if pvals[idx_to_remove] < threshold:
            break

        print(f"Removing GAM term index {idx_to_remove}, p-value = {pvals[idx_to_remove]:.6f}")
        del terms_work[idx_to_remove]

    if len(terms_work) == 0:
        return None, []

    try:
        gam_final = gam_class(reduce(operator.add, terms_work)).fit(X, y)
        return gam_final, terms_work
    except Exception as e:
        print(f"[Stepwise-GAM] final fit failed: {e}")
        return None, []


# =========================
# 3. Model builders
# =========================
def build_regression_models():
    models = [
        ('XGBoost', XGBRegressor(
            random_state=SEED,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method='hist',
            device='cpu',
            n_jobs=4,
        )),
        ('Random Forest', RandomForestRegressor(
            random_state=SEED,
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=4,
        )),
        ('Ridge', make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ('SVR', make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1, kernel='rbf'))),
        ('AdaBoost', AdaBoostRegressor(
            random_state=SEED,
            n_estimators=300,
            learning_rate=0.05,
        )),
    ]

    if HAS_LIGHTGBM:
        models.append(('LightGBM', LGBMRegressor(
            random_state=SEED,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            verbose=-1,
        )))
    else:
        print('[Warning] lightgbm is not installed. LightGBM regressor will be skipped.')

    if HAS_CATBOOST:
        models.append(('CatBoost', CatBoostRegressor(
            random_state=SEED,
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function='RMSE',
            verbose=False,
        )))
    else:
        print('[Warning] catboost is not installed. CatBoost regressor will be skipped.')

    return models


def build_classification_models():
    models = [
        ('XGBoost', XGBClassifier(
            random_state=SEED,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            tree_method='hist',
            device='cpu',
            n_jobs=4,
        )),
        ('Random Forest', RandomForestClassifier(
            random_state=SEED,
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=4,
        )),
        ('RidgeClassifier', make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))),
        ('SVC', make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf'))),
        ('AdaBoost', AdaBoostClassifier(
            random_state=SEED,
            n_estimators=300,
            learning_rate=0.05,
        )),
    ]

    if HAS_LIGHTGBM:
        models.append(('LightGBM', LGBMClassifier(
            random_state=SEED,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            verbose=-1,
        )))
    else:
        print('[Warning] lightgbm is not installed. LightGBM classifier will be skipped.')

    if HAS_CATBOOST:
        models.append(('CatBoost', CatBoostClassifier(
            random_state=SEED,
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function='Logloss',
            verbose=False,
        )))
    else:
        print('[Warning] catboost is not installed. CatBoost classifier will be skipped.')

    return models


def safe_predict_binary(model_name, model, X):
    if model_name == 'Stepwise-LogisticGAM':
        proba = np.asarray(model.predict_proba(X)).reshape(-1)
        pred = (proba >= 0.5).astype(int)
        return pred, proba

    pred = np.asarray(model.predict(X)).reshape(-1).astype(int)

    if hasattr(model, 'predict_proba'):
        try:
            proba = np.asarray(model.predict_proba(X))[:, 1]
        except Exception:
            proba = np.full(len(pred), np.nan)
    elif hasattr(model, 'decision_function'):
        try:
            raw = np.asarray(model.decision_function(X)).reshape(-1)
            proba = 1 / (1 + np.exp(-raw))
        except Exception:
            proba = np.full(len(pred), np.nan)
    else:
        proba = np.full(len(pred), np.nan)

    return pred, proba


# =========================
# 4. LOCO regression
# =========================
def run_regression_loco(df):
    per_country_metric_rows = []
    pred_rows = []
    gam_spline_rows = []
    all_model_predictions = {}

    countries = sorted(df[COUNTRY_COL].unique().tolist())

    for held_out_country in countries:
        train_df = df[df[COUNTRY_COL] != held_out_country].copy()
        valid_df = df[df[COUNTRY_COL] == held_out_country].copy()

        X_train = train_df[FEATURES].values
        y_train = train_df[TARGET].values
        X_valid = valid_df[FEATURES].values
        y_valid = valid_df[TARGET].values

        print('\n' + '=' * 80)
        print(f'[Regression] Held-out country: {held_out_country}')
        print(f'Train n = {len(train_df)}, Valid n = {len(valid_df)}')

        reg_terms, reg_k_dict = choose_best_terms(X_train, y_train, FEATURES, LinearGAM, k_range=range(4, 9))
        gam_reg, _ = stepwise_gam_fit(X_train, y_train, reg_terms, LinearGAM, threshold=0.05)

        for term_name, best_k in reg_k_dict.items():
            gam_spline_rows.append({
                'HeldOutCountry': held_out_country,
                'Term': term_name,
                'Best_n_splines': best_k,
            })

        models = []
        if gam_reg is not None:
            models.append(('Stepwise-GAM', gam_reg))
        else:
            print(f'[Warning] Stepwise-GAM failed for fold {held_out_country} and will be skipped.')

        models.extend(build_regression_models())

        for model_name, model in models:
            print(f'[Regression] Fitting {model_name} on fold {held_out_country}...')
            try:
                if model_name != 'Stepwise-GAM':
                    model.fit(X_train, y_train)

                pred_valid = np.asarray(model.predict(X_valid)).reshape(-1)
                fold_metrics = get_regression_metrics(y_valid, pred_valid)

                per_country_metric_rows.append({
                    'HeldOutCountry': held_out_country,
                    'Model': model_name,
                    'Train_n': len(train_df),
                    'Valid_n': len(valid_df),
                    **fold_metrics,
                })

                pred_df = pd.DataFrame({
                    'HeldOutCountry': held_out_country,
                    'Model': model_name,
                    'country': valid_df[COUNTRY_COL].values,
                    'Observed_R': y_valid,
                    'Predicted_R': pred_valid,
                })
                pred_rows.append(pred_df)
                all_model_predictions.setdefault(model_name, []).append(pred_df)

            except Exception as e:
                print(f'[Warning] {model_name} failed on fold {held_out_country}: {e}')
                continue

    per_country_metrics_df = pd.DataFrame(per_country_metric_rows)
    predictions_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    gam_splines_df = pd.DataFrame(gam_spline_rows)

    overall_rows = []
    for model_name, parts in all_model_predictions.items():
        pooled = pd.concat(parts, ignore_index=True)
        overall_metrics = get_regression_metrics(pooled['Observed_R'].values, pooled['Predicted_R'].values)
        overall_rows.append({
            'Model': model_name,
            'Total_n': len(pooled),
            **overall_metrics,
        })
    overall_metrics_df = pd.DataFrame(overall_rows)

    summary_df = summarize_fold_metrics(
        per_country_metrics_df,
        metric_cols=['RMSE', 'MAE', 'R2', 'Pearson_r'],
        task_name='Regression'
    )

    if not per_country_metrics_df.empty:
        per_country_metrics_df = per_country_metrics_df.sort_values(['Model', 'HeldOutCountry']).reset_index(drop=True)
    if not overall_metrics_df.empty:
        overall_metrics_df = overall_metrics_df.sort_values(['Model']).reset_index(drop=True)
    if not predictions_df.empty:
        predictions_df = predictions_df.sort_values(['Model', 'HeldOutCountry']).reset_index(drop=True)
    if not gam_splines_df.empty:
        gam_splines_df = gam_splines_df.sort_values(['HeldOutCountry', 'Term']).reset_index(drop=True)

    return per_country_metrics_df, summary_df, overall_metrics_df, predictions_df, gam_splines_df


# =========================
# 5. LOCO classification
# =========================
def run_classification_loco(df):
    per_country_metric_rows = []
    pred_rows = []
    gam_spline_rows = []
    all_model_predictions = {}

    countries = sorted(df[COUNTRY_COL].unique().tolist())

    for held_out_country in countries:
        train_df = df[df[COUNTRY_COL] != held_out_country].copy()
        valid_df = df[df[COUNTRY_COL] == held_out_country].copy()

        X_train = train_df[FEATURES].values
        y_train = train_df['fitness_class'].values
        X_valid = valid_df[FEATURES].values
        y_valid = valid_df['fitness_class'].values

        print('\n' + '=' * 80)
        print(f'[Classification] Held-out country: {held_out_country}')
        print(f'Train n = {len(train_df)}, Valid n = {len(valid_df)}')
        print(f'Train positives = {int(y_train.sum())}, Valid positives = {int(y_valid.sum())}')

        if len(np.unique(y_train)) < 2:
            print(f'[Skip fold] Training set for {held_out_country} has only one class.')
            continue

        cls_terms, cls_k_dict = choose_best_terms(X_train, y_train, FEATURES, LogisticGAM, k_range=range(4, 9))
        gam_cls, _ = stepwise_gam_fit(X_train, y_train, cls_terms, LogisticGAM, threshold=0.05)

        for term_name, best_k in cls_k_dict.items():
            gam_spline_rows.append({
                'HeldOutCountry': held_out_country,
                'Term': term_name,
                'Best_n_splines': best_k,
            })

        models = []
        if gam_cls is not None:
            models.append(('Stepwise-LogisticGAM', gam_cls))
        else:
            print(f'[Warning] Stepwise-LogisticGAM failed for fold {held_out_country} and will be skipped.')

        models.extend(build_classification_models())

        for model_name, model in models:
            print(f'[Classification] Fitting {model_name} on fold {held_out_country}...')
            try:
                if model_name != 'Stepwise-LogisticGAM':
                    model.fit(X_train, y_train)

                pred_valid, proba_valid = safe_predict_binary(model_name, model, X_valid)
                fold_metrics = get_classification_metrics(y_valid, pred_valid)

                per_country_metric_rows.append({
                    'HeldOutCountry': held_out_country,
                    'Model': model_name,
                    'Train_n': len(train_df),
                    'Valid_n': len(valid_df),
                    'Train_Positive_n': int(y_train.sum()),
                    'Valid_Positive_n': int(y_valid.sum()),
                    **fold_metrics,
                })

                pred_df = pd.DataFrame({
                    'HeldOutCountry': held_out_country,
                    'Model': model_name,
                    'country': valid_df[COUNTRY_COL].values,
                    'Observed_R': valid_df[TARGET].values,
                    'Observed_Class': y_valid,
                    'Predicted_Class': pred_valid,
                    'Predicted_Prob': proba_valid,
                })
                pred_rows.append(pred_df)
                all_model_predictions.setdefault(model_name, []).append(pred_df)

            except Exception as e:
                print(f'[Warning] {model_name} failed on fold {held_out_country}: {e}')
                continue

    per_country_metrics_df = pd.DataFrame(per_country_metric_rows)
    predictions_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    gam_splines_df = pd.DataFrame(gam_spline_rows)

    overall_rows = []
    for model_name, parts in all_model_predictions.items():
        pooled = pd.concat(parts, ignore_index=True)
        overall_metrics = get_classification_metrics(pooled['Observed_Class'].values, pooled['Predicted_Class'].values)
        overall_rows.append({
            'Model': model_name,
            'Total_n': len(pooled),
            'Total_Positive_n': int(pooled['Observed_Class'].sum()),
            **overall_metrics,
        })
    overall_metrics_df = pd.DataFrame(overall_rows)

    summary_df = summarize_fold_metrics(
        per_country_metrics_df,
        metric_cols=['Accuracy', 'Precision', 'Recall', 'F1'],
        task_name='Classification'
    )

    if not per_country_metrics_df.empty:
        per_country_metrics_df = per_country_metrics_df.sort_values(['Model', 'HeldOutCountry']).reset_index(drop=True)
    if not overall_metrics_df.empty:
        overall_metrics_df = overall_metrics_df.sort_values(['Model']).reset_index(drop=True)
    if not predictions_df.empty:
        predictions_df = predictions_df.sort_values(['Model', 'HeldOutCountry']).reset_index(drop=True)
    if not gam_splines_df.empty:
        gam_splines_df = gam_splines_df.sort_values(['HeldOutCountry', 'Term']).reset_index(drop=True)

    return per_country_metrics_df, summary_df, overall_metrics_df, predictions_df, gam_splines_df


# =========================
# 6. Main
# =========================
def main():
    output_dir = get_output_dir()
    print(f'Current working directory: {os.getcwd()}')
    print(f'Output directory: {output_dir}')

    data = pd.read_csv(DATA_PATH)
    data = prepare_data(data)

    print('\n=== Cleaned sample counts ===')
    print(f'Total cleaned rows (9 countries only): {len(data)}')
    print('\nRows by country:')
    print(data[COUNTRY_COL].value_counts().sort_index())

    print('\nPositive class counts by country (R > CLASS_CUTOFF):')
    print(
        data.groupby(COUNTRY_COL)['fitness_class']
        .agg(['sum', 'count'])
        .rename(columns={'sum': 'positive_n', 'count': 'total_n'})
        .sort_index()
    )

    # Regression
    reg_by_country, reg_summary, reg_overall, reg_preds, reg_gam = run_regression_loco(data)

    print('\n=== Regression: by-country metrics ===')
    print(reg_by_country.round(4) if not reg_by_country.empty else 'No regression results.')
    print('\n=== Regression: summary (mean/std/95% CI across countries) ===')
    print(reg_summary.round(4) if not reg_summary.empty else 'No regression summary.')
    print('\n=== Regression: overall pooled metrics ===')
    print(reg_overall.round(4) if not reg_overall.empty else 'No regression overall metrics.')

    # Classification
    cls_by_country, cls_summary, cls_overall, cls_preds, cls_gam = run_classification_loco(data)

    print('\n=== Classification: by-country metrics ===')
    print(cls_by_country.round(4) if not cls_by_country.empty else 'No classification results.')
    print('\n=== Classification: summary (mean/std/95% CI across countries) ===')
    print(cls_summary.round(4) if not cls_summary.empty else 'No classification summary.')
    print('\n=== Classification: overall pooled metrics ===')
    print(cls_overall.round(4) if not cls_overall.empty else 'No classification overall metrics.')

    # Save all outputs
    reg_by_country.to_csv(os.path.join(output_dir, 'regression_metrics_by_country_loco.csv'), index=False)
    reg_summary.to_csv(os.path.join(output_dir, 'regression_metrics_summary_loco.csv'), index=False)
    reg_overall.to_csv(os.path.join(output_dir, 'regression_metrics_overall_loco.csv'), index=False)
    reg_preds.to_csv(os.path.join(output_dir, 'regression_predictions_loco.csv'), index=False)
    reg_gam.to_csv(os.path.join(output_dir, 'regression_gam_best_splines_loco.csv'), index=False)

    cls_by_country.to_csv(os.path.join(output_dir, 'classification_metrics_by_country_loco.csv'), index=False)
    cls_summary.to_csv(os.path.join(output_dir, 'classification_metrics_summary_loco.csv'), index=False)
    cls_overall.to_csv(os.path.join(output_dir, 'classification_metrics_overall_loco.csv'), index=False)
    cls_preds.to_csv(os.path.join(output_dir, 'classification_predictions_loco.csv'), index=False)
    cls_gam.to_csv(os.path.join(output_dir, 'classification_gam_best_splines_loco.csv'), index=False)

    print('\nSaved files under:')
    print(output_dir)
    for name in [
        'regression_metrics_by_country_loco.csv',
        'regression_metrics_summary_loco.csv',
        'regression_metrics_overall_loco.csv',
        'regression_predictions_loco.csv',
        'regression_gam_best_splines_loco.csv',
        'classification_metrics_by_country_loco.csv',
        'classification_metrics_summary_loco.csv',
        'classification_metrics_overall_loco.csv',
        'classification_predictions_loco.csv',
        'classification_gam_best_splines_loco.csv',
    ]:
        print(f'- {name}')


if __name__ == '__main__':
    main()
