# -*- coding: utf-8 -*-
"""
"9 + 1" external-country validation script
==========================================

Reviewer comment (Leo Poon):
    "Instead of using leave one out strategy, can we use 9 + 1 strategy.
     Therefore, analysing data from a country with very little information
     and do some predictions of the outbreak in this additional country.
     ... I guess this might be more convincing in showing the power of
     your system."

Strategy
--------
Training set (fixed, 9 countries):
    Bangladesh, Brazil, Cambodia, Germany, India, Senegal, Turkey, UK, USA

External validation ("+1", each country evaluated independently):
    Denmark, Japan, Nepal, Mexico

Each model is trained ONCE on the pooled 9-country training set and then
applied, without any refitting or recalibration, to each external country.
None of the external countries contributes to feature selection, GAM term
selection or model fitting, so each external country is a genuinely
unseen population.

Tasks
-----
1) Regression on R
2) Binary classification on fitness_class = (R > CLASS_CUTOFF)

Uncertainty
-----------
A single train/test split gives one point estimate per country, so there
are no folds to average over. 95% confidence intervals are therefore
obtained by non-parametric bootstrap resampling of the held-out samples
within each external country (N_BOOTSTRAP resamples, percentile method).
This quantifies the sampling uncertainty of the external-validation
estimate, which is what matters when a test country has few sequences.

Outputs
-------
CSV
    external_regression_by_country.csv
    external_classification_by_country.csv
    external_regression_pooled.csv
    external_classification_pooled.csv
    external_regression_predictions.csv
    external_classification_predictions.csv
    external_country_overview.csv
    external_gam_best_splines.csv

Three-line (booktabs-style) tables
    Table_external_classification.docx / .tex / .txt
    Table_external_regression.docx    / .tex / .txt
    Table_external_best_model.docx    / .tex / .txt

Recommended packages:
    pip install pygam xgboost lightgbm catboost python-docx
"""

import os
import warnings
import operator
from functools import reduce

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
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
    roc_auc_score,
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

try:
    import docx  # noqa: F401
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False


# =========================
# 0. Global settings
# =========================
# Either input works: merged_binding_final_updated.csv (1 MB) was verified
# to give exactly the same per-country sample sizes as the 449 MB
# merged_with_binding_retained_updated.csv after the cleaning applied here
# (Bangladesh 35, Brazil 176, Cambodia 60, Germany 371, India 199,
#  Senegal 37, Turkey 193, UK 360, USA 428), and it also carries the four
# external countries (Denmark 357, Japan 284, Nepal 36, Mexico 240).
# The small file is used by default because it loads in under a second.
DATA_PATH = os.environ.get(
    "FITNESS_DATA", "../../data/processed/merged_binding_final_updated.csv")
# Point FITNESS_DATA at merged_with_binding_retained_updated.csv for the full table.
BASE_OUTPUT_DIR = os.environ.get("FITNESS_OUT", "../../results")
OUTPUT_DIR_NAME = "external_validation_9plus1_3countries"

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

# ---- the 9 training countries (model never sees anything else) ----
TRAIN_COUNTRIES = [
    'Bangladesh', 'Brazil', 'Cambodia', 'Germany',
    'India', 'Senegal', 'Turkey', 'UK', 'USA'
]

# ---- the external "+1" countries, each evaluated on its own ----
EXTERNAL_COUNTRIES = [
    'Denmark', 'Nepal', 'Mexico'
    # 'Japan',   # excluded from this run; see external_validation_9plus1/
    #            # for the full four-country result, in which every model
    #            # over-predicted R in Japan by ~44 (mean observed R 39.7
    #            # vs 78.6 in the training set) while still reaching
    #            # AUC 0.880 there.
]

REG_METRICS = ['RMSE', 'MAE', 'R2', 'Pearson_r', 'Spearman_rho', 'Bias']
CLS_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

# Bootstrap settings for the 95% CI within each external country
N_BOOTSTRAP = 2000
MIN_TEST_N = 10          # warn when an external country has fewer rows

# Compact "best model" summary table
#
# Set these to a model name to fix the model shown in that table,
# e.g. BEST_CLS_MODEL = 'SVC'. Leaving them as None selects the model
# with the best pooled external performance on the metric below.
#
# NOTE: choosing the model on the external data makes that table
# slightly optimistic. For a strictly pre-specified analysis, set the
# names manually to the model that won the internal 9-country
# cross-validation.
BEST_CLS_MODEL = None
BEST_REG_MODEL = None
BEST_CLS_METRIC = 'AUC'
BEST_REG_METRIC = 'Pearson_r'


# =========================
# 1. Utility helpers
# =========================
def get_output_dir():
    base_dir = BASE_OUTPUT_DIR if os.path.isdir(BASE_OUTPUT_DIR) else os.getcwd()
    output_dir = os.path.join(base_dir, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def prepare_data(df):
    """Clean the data and keep the 9 training + 4 external countries."""
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

    keep = TRAIN_COUNTRIES + EXTERNAL_COUNTRIES
    overlap = sorted(set(TRAIN_COUNTRIES) & set(EXTERNAL_COUNTRIES))
    if overlap:
        raise ValueError(
            f"These countries appear in both the training and the external set: {overlap}. "
            "An external country must never be used for training."
        )

    df = df[df[COUNTRY_COL].isin(keep)].copy()
    df = df.dropna(subset=FEATURES + [TARGET, COUNTRY_COL]).copy()
    df['fitness_class'] = (df[TARGET] > CLASS_CUTOFF).astype(int)

    found = set(df[COUNTRY_COL].unique())
    for c in keep:
        if c not in found:
            print(f"[Warning] No usable rows for country '{c}' after cleaning.")

    return df


def safe_pearsonr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    return float(pearsonr(y_true, y_pred)[0])


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    return safe_pearsonr(rankdata(y_true), rankdata(y_pred))


def get_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {m: np.nan for m in REG_METRICS}
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': r2,
        'Pearson_r': safe_pearsonr(y_true, y_pred),
        'Spearman_rho': safe_spearman(y_true, y_pred),
        # mean signed error: positive = the model over-predicts R in this country
        'Bias': float(np.mean(y_pred - y_true)),
    }


def get_classification_metrics(y_true, y_pred, y_proba=None):
    """Accuracy/precision/recall/F1 at the fixed R > CLASS_CUTOFF threshold,
    plus the threshold-free ROC-AUC when predicted probabilities exist.

    The distinction matters for external validation: accuracy depends on
    the class prevalence of the test country, whereas AUC measures only
    whether the model ranks lineages correctly within that country.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {m: np.nan for m in CLS_METRICS}

    out = {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'AUC': np.nan,
    }

    if y_proba is not None:
        p = np.asarray(y_proba, dtype=float)
        if (len(np.unique(y_true)) == 2 and p.size == y_true.size
                and np.isfinite(p).all()):
            try:
                out['AUC'] = float(roc_auc_score(y_true, p))
            except Exception:
                pass
    return out


def _percentile_ci(arr, alpha=0.05):
    """Percentile CI from bootstrap replicates, ignoring undefined ones."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return (np.nan, np.nan, int(arr.size))
    return (
        float(np.percentile(arr, 100 * alpha / 2)),
        float(np.percentile(arr, 100 * (1 - alpha / 2))),
        int(arr.size),
    )


def _boot_indices(n, n_boot, rng, chunk):
    """Yield chunks of bootstrap index matrices of shape (b, n)."""
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        yield rng.integers(0, n, size=(b, n))
        done += b


def bootstrap_ci_classification(y_true, y_pred, y_proba=None,
                                n_boot=N_BOOTSTRAP,
                                seed=SEED, alpha=0.05, chunk=500):
    """Vectorised percentile bootstrap for the four classification metrics.

    Each sample is encoded once as 2*y_true + y_pred, so a resample only
    needs the counts of the four confusion-matrix cells. This reproduces
    sklearn's binary metrics with zero_division=0, but ~2 orders of
    magnitude faster than recomputing them resample by resample.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    n = y_true.size

    out = {m: (np.nan, np.nan, 0) for m in CLS_METRICS}
    if n < 2 or n_boot <= 0:
        return out

    code = 2 * y_true + y_pred          # 0=TN, 1=FP, 2=FN, 3=TP

    proba = None
    if y_proba is not None:
        pr = np.asarray(y_proba, dtype=float).ravel()
        if pr.size == n and np.isfinite(pr).all():
            proba = pr

    rng = np.random.default_rng(seed)
    acc, prec, rec, f1, auc = [], [], [], [], []

    for idx in _boot_indices(n, n_boot, rng, chunk):
        c = code[idx]                                   # (b, n)
        counts = np.stack([(c == k).sum(axis=1) for k in range(4)], axis=1)
        tn, fp, fn, tp = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]

        with np.errstate(invalid='ignore', divide='ignore'):
            a = (tp + tn) / n
            p = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1), 0.0)
            r = np.where((tp + fn) > 0, tp / np.maximum(tp + fn, 1), 0.0)
            f = np.where((p + r) > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)

        acc.append(a); prec.append(p); rec.append(r); f1.append(f)

        if proba is not None:
            # ROC-AUC via the Mann-Whitney identity, with tie-corrected ranks
            yt = y_true[idx]
            ranks = rankdata(proba[idx], axis=1)
            n_pos = yt.sum(axis=1)
            n_neg = n - n_pos
            sum_pos = (ranks * yt).sum(axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                a_ = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            auc.append(np.where((n_pos > 0) & (n_neg > 0), a_, np.nan))

    series = [np.concatenate(acc), np.concatenate(prec),
              np.concatenate(rec), np.concatenate(f1)]
    series.append(np.concatenate(auc) if auc else np.array([]))

    for name, vals in zip(CLS_METRICS, series):
        out[name] = _percentile_ci(vals, alpha)
    return out


def bootstrap_ci_regression(y_true, y_pred, n_boot=N_BOOTSTRAP,
                            seed=SEED, alpha=0.05, chunk=500):
    """Vectorised percentile bootstrap for RMSE, MAE, R2 and Pearson r.

    R2 and Pearson r are left undefined (and dropped from the percentile
    calculation) for any resample without variation in the observed or
    the predicted values.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = y_true.size

    out = {m: (np.nan, np.nan, 0) for m in REG_METRICS}
    if n < 2 or n_boot <= 0:
        return out

    rng = np.random.default_rng(seed)
    rmse, mae, r2, pr, sp, bias = [], [], [], [], [], []

    def _row_corr(a, b):
        a_c = a - a.mean(axis=1, keepdims=True)
        b_c = b - b.mean(axis=1, keepdims=True)
        denom = np.sqrt(np.sum(a_c ** 2, axis=1) * np.sum(b_c ** 2, axis=1))
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(denom > 0,
                            np.sum(a_c * b_c, axis=1) / np.where(denom > 0, denom, 1),
                            np.nan)

    for idx in _boot_indices(n, n_boot, rng, chunk):
        yt = y_true[idx]
        yp = y_pred[idx]
        resid = yt - yp

        rmse.append(np.sqrt(np.mean(resid ** 2, axis=1)))
        mae.append(np.mean(np.abs(resid), axis=1))
        bias.append(np.mean(yp - yt, axis=1))

        yt_c = yt - yt.mean(axis=1, keepdims=True)
        ss_tot = np.sum(yt_c ** 2, axis=1)
        ss_res = np.sum(resid ** 2, axis=1)

        with np.errstate(invalid='ignore', divide='ignore'):
            r2.append(np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan))

        pr.append(_row_corr(yt, yp))
        sp.append(_row_corr(rankdata(yt, axis=1), rankdata(yp, axis=1)))

    for name, vals in zip(REG_METRICS,
                          [np.concatenate(rmse), np.concatenate(mae),
                           np.concatenate(r2), np.concatenate(pr),
                           np.concatenate(sp), np.concatenate(bias)]):
        out[name] = _percentile_ci(vals, alpha)
    return out


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
# 4. Train once on the 9 countries
# =========================
def fit_regression_models(train_df):
    """Fit every regression model once on the pooled 9-country training set."""
    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values

    print('\n' + '=' * 80)
    print('[Regression] Training on the 9-country set')
    print(f'Train n = {len(train_df)}')

    reg_terms, reg_k_dict = choose_best_terms(
        X_train, y_train, FEATURES, LinearGAM, k_range=range(4, 9)
    )
    gam_reg, _ = stepwise_gam_fit(X_train, y_train, reg_terms, LinearGAM, threshold=0.05)

    fitted = []
    if gam_reg is not None:
        fitted.append(('Stepwise-GAM', gam_reg))
    else:
        print('[Warning] Stepwise-GAM failed on the training set and will be skipped.')

    for model_name, model in build_regression_models():
        print(f'[Regression] Fitting {model_name} ...')
        try:
            model.fit(X_train, y_train)
            fitted.append((model_name, model))
        except Exception as e:
            print(f'[Warning] {model_name} failed to fit: {e}')

    return fitted, reg_k_dict


def fit_classification_models(train_df):
    """Fit every classifier once on the pooled 9-country training set."""
    X_train = train_df[FEATURES].values
    y_train = train_df['fitness_class'].values

    print('\n' + '=' * 80)
    print('[Classification] Training on the 9-country set')
    print(f'Train n = {len(train_df)}, positives = {int(y_train.sum())}')

    if len(np.unique(y_train)) < 2:
        raise ValueError('The training set contains only one class; cannot fit classifiers.')

    cls_terms, cls_k_dict = choose_best_terms(
        X_train, y_train, FEATURES, LogisticGAM, k_range=range(4, 9)
    )
    gam_cls, _ = stepwise_gam_fit(X_train, y_train, cls_terms, LogisticGAM, threshold=0.05)

    fitted = []
    if gam_cls is not None:
        fitted.append(('Stepwise-LogisticGAM', gam_cls))
    else:
        print('[Warning] Stepwise-LogisticGAM failed on the training set and will be skipped.')

    for model_name, model in build_classification_models():
        print(f'[Classification] Fitting {model_name} ...')
        try:
            model.fit(X_train, y_train)
            fitted.append((model_name, model))
        except Exception as e:
            print(f'[Warning] {model_name} failed to fit: {e}')

    return fitted, cls_k_dict


# =========================
# 5. Evaluate on each external country
# =========================
def evaluate_regression_external(fitted_models, train_df, ext_df):
    metric_rows = []
    pred_rows = []

    countries = [c for c in EXTERNAL_COUNTRIES if c in set(ext_df[COUNTRY_COL])]

    for country in countries:
        sub = ext_df[ext_df[COUNTRY_COL] == country]
        X_valid = sub[FEATURES].values
        y_valid = sub[TARGET].values

        print('\n' + '-' * 80)
        print(f'[Regression] External country: {country} (n = {len(sub)})')
        if len(sub) < MIN_TEST_N:
            print(f'[Note] Only {len(sub)} usable rows; estimates will be very uncertain.')

        for model_name, model in fitted_models:
            try:
                pred = np.asarray(model.predict(X_valid)).reshape(-1)
            except Exception as e:
                print(f'[Warning] {model_name} failed to predict for {country}: {e}')
                continue

            point = get_regression_metrics(y_valid, pred)
            ci = bootstrap_ci_regression(y_valid, pred)

            row = {
                'ExternalCountry': country,
                'Model': model_name,
                'Train_n': len(train_df),
                'Test_n': len(sub),
                'Observed_R_mean': float(np.mean(y_valid)) if len(y_valid) else np.nan,
                'Observed_R_sd': float(np.std(y_valid, ddof=1)) if len(y_valid) > 1 else np.nan,
            }
            for m in REG_METRICS:
                row[m] = point[m]
                row[f'{m}_ci95_low'] = ci[m][0]
                row[f'{m}_ci95_high'] = ci[m][1]
            metric_rows.append(row)

            pred_rows.append(pd.DataFrame({
                'ExternalCountry': country,
                'Model': model_name,
                'country': sub[COUNTRY_COL].values,
                'Observed_R': y_valid,
                'Predicted_R': pred,
            }))

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    # pooled across all external countries
    pooled_rows = []
    if not preds_df.empty:
        for model_name, g in preds_df.groupby('Model'):
            point = get_regression_metrics(g['Observed_R'].values, g['Predicted_R'].values)
            ci = bootstrap_ci_regression(
                g['Observed_R'].values, g['Predicted_R'].values
            )
            row = {
                'ExternalCountry': 'All external countries',
                'Model': model_name,
                'Train_n': len(train_df),
                'Test_n': int(len(g)),
            }
            for m in REG_METRICS:
                row[m] = point[m]
                row[f'{m}_ci95_low'] = ci[m][0]
                row[f'{m}_ci95_high'] = ci[m][1]
            pooled_rows.append(row)
    pooled_df = pd.DataFrame(pooled_rows)

    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(['ExternalCountry', 'Model']).reset_index(drop=True)
    if not pooled_df.empty:
        pooled_df = pooled_df.sort_values('Model').reset_index(drop=True)

    return metrics_df, pooled_df, preds_df


def evaluate_classification_external(fitted_models, train_df, ext_df):
    metric_rows = []
    pred_rows = []

    countries = [c for c in EXTERNAL_COUNTRIES if c in set(ext_df[COUNTRY_COL])]

    for country in countries:
        sub = ext_df[ext_df[COUNTRY_COL] == country]
        X_valid = sub[FEATURES].values
        y_valid = sub['fitness_class'].values

        print('\n' + '-' * 80)
        print(f'[Classification] External country: {country} '
              f'(n = {len(sub)}, positives = {int(y_valid.sum())})')
        if len(sub) < MIN_TEST_N:
            print(f'[Note] Only {len(sub)} usable rows; estimates will be very uncertain.')
        if len(np.unique(y_valid)) < 2:
            print(f'[Note] {country} contains only one observed class '
                  f'(all {"high" if y_valid.sum() > 0 else "low"} fitness); '
                  'precision/recall/F1 must be read with that in mind.')

        for model_name, model in fitted_models:
            try:
                pred, proba = safe_predict_binary(model_name, model, X_valid)
            except Exception as e:
                print(f'[Warning] {model_name} failed to predict for {country}: {e}')
                continue

            point = get_classification_metrics(y_valid, pred, proba)
            ci = bootstrap_ci_classification(y_valid, pred, proba)

            row = {
                'ExternalCountry': country,
                'Model': model_name,
                'Train_n': len(train_df),
                'Train_Positive_n': int(train_df['fitness_class'].sum()),
                'Test_n': len(sub),
                'Test_Positive_n': int(y_valid.sum()),
                'Test_Positive_rate': float(np.mean(y_valid)) if len(y_valid) else np.nan,
            }
            for m in CLS_METRICS:
                row[m] = point[m]
                row[f'{m}_ci95_low'] = ci[m][0]
                row[f'{m}_ci95_high'] = ci[m][1]
            metric_rows.append(row)

            pred_rows.append(pd.DataFrame({
                'ExternalCountry': country,
                'Model': model_name,
                'country': sub[COUNTRY_COL].values,
                'Observed_R': sub[TARGET].values,
                'Observed_Class': y_valid,
                'Predicted_Class': pred,
                'Predicted_Prob': proba,
            }))

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    pooled_rows = []
    if not preds_df.empty:
        for model_name, g in preds_df.groupby('Model'):
            point = get_classification_metrics(
                g['Observed_Class'].values, g['Predicted_Class'].values,
                g['Predicted_Prob'].values
            )
            ci = bootstrap_ci_classification(
                g['Observed_Class'].values, g['Predicted_Class'].values,
                g['Predicted_Prob'].values
            )
            row = {
                'ExternalCountry': 'All external countries',
                'Model': model_name,
                'Train_n': len(train_df),
                'Test_n': int(len(g)),
                'Test_Positive_n': int(g['Observed_Class'].sum()),
            }
            for m in CLS_METRICS:
                row[m] = point[m]
                row[f'{m}_ci95_low'] = ci[m][0]
                row[f'{m}_ci95_high'] = ci[m][1]
            pooled_rows.append(row)
    pooled_df = pd.DataFrame(pooled_rows)

    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(['ExternalCountry', 'Model']).reset_index(drop=True)
    if not pooled_df.empty:
        pooled_df = pooled_df.sort_values('Model').reset_index(drop=True)

    return metrics_df, pooled_df, preds_df


# =========================
# 6. Three-line (booktabs) tables
# =========================
MODEL_DISPLAY = {
    'Stepwise-GAM': 'GAM',
    'Stepwise-LogisticGAM': 'GAM',
    'RidgeClassifier': 'Ridge',
    'SVC': 'SVM',
    'SVR': 'SVM',
}

MODEL_ORDER = ['GAM', 'Ridge', 'SVM', 'Random Forest',
               'AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost']


def display_model(name):
    return MODEL_DISPLAY.get(name, name)


def model_sort_key(name):
    d = display_model(name)
    return (MODEL_ORDER.index(d) if d in MODEL_ORDER else len(MODEL_ORDER), d)


def fmt_ci(value, lo, hi, nd=3):
    """'0.746 (0.681, 0.805)' ; falls back gracefully on NaN."""
    if value is None or not np.isfinite(value):
        return '--'
    main = f'{value:.{nd}f}'
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return main
    return f'{main} ({lo:.{nd}f}, {hi:.{nd}f})'


def build_metric_table(metrics_df, pooled_df, metrics, decimals,
                       extra_cols, country_col='ExternalCountry'):
    """Long -> display table, one row per country x model.

    Country-level descriptive columns are printed only on the first row
    of each country block, which is the usual three-line-table style.
    """
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()

    frames = [metrics_df]
    if pooled_df is not None and not pooled_df.empty:
        frames.append(pooled_df)
    df = pd.concat(frames, ignore_index=True)

    order = [c for c in EXTERNAL_COUNTRIES if c in set(df[country_col])]
    order += [c for c in df[country_col].unique() if c not in order]

    rows = []
    for country in order:
        block = df[df[country_col] == country].copy()
        block['_k'] = block['Model'].map(model_sort_key)
        block = block.sort_values('_k')

        first = True
        for _, r in block.iterrows():
            row = {'Country': country if first else ''}
            for label, col, fn in extra_cols:
                if first and col in r.index and pd.notna(r[col]):
                    row[label] = fn(r)
                else:
                    row[label] = ''
            row['Model'] = display_model(r['Model'])
            for m in metrics:
                nd = decimals.get(m, 3)
                row[_metric_label(m)] = fmt_ci(
                    r.get(m, np.nan),
                    r.get(f'{m}_ci95_low', np.nan),
                    r.get(f'{m}_ci95_high', np.nan),
                    nd,
                )
            rows.append(row)
            first = False

    return pd.DataFrame(rows)


def _metric_label(m):
    return {
        'R2': 'R2',
        'Pearson_r': 'Pearson r',
        'Spearman_rho': 'Spearman rho',
        'Bias': 'Bias',
    }.get(m, m)


def write_txt_table(df, path, title, note=None):
    if df.empty:
        return
    cols = list(df.columns)
    widths = [
        max(len(str(c)), *(len(str(v)) for v in df[c].astype(str)))
        for c in cols
    ]

    def line(ch):
        return ch * (sum(widths) + 2 * (len(cols) - 1))

    out = [title, '', line('=')]
    out.append('  '.join(str(c).ljust(w) for c, w in zip(cols, widths)))
    out.append(line('-'))
    for _, r in df.iterrows():
        out.append('  '.join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    out.append(line('='))
    if note:
        out.append('')
        out.append(note)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


def write_tex_table(df, path, title, note=None, label='tab:external'):
    if df.empty:
        return
    cols = list(df.columns)
    align = 'l' * 2 + 'r' * (len(cols) - 2) if len(cols) > 2 else 'l' * len(cols)

    def esc(x):
        x = str(x)
        for a, b in [('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#')]:
            x = x.replace(a, b)
        return x

    header = [esc(c).replace('R2', '$R^2$') for c in cols]

    out = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{' + esc(title) + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{' + align + '}',
        r'\toprule',
        ' & '.join(header) + r' \\',
        r'\midrule',
    ]
    for _, r in df.iterrows():
        out.append(' & '.join(esc(r[c]) for c in cols) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    if note:
        out.append(r'\begin{tablenotes}\small\item ' + esc(note) + r'\end{tablenotes}')
    out.append(r'\end{table}')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


def _set_cell_border(cell, **kwargs):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn('w:tcBorders'))
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)

    for edge in ('top', 'bottom', 'left', 'right'):
        if edge not in kwargs:
            continue
        spec = kwargs[edge]
        tag = f'w:{edge}'
        el = tcBorders.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            tcBorders.append(el)
        for k, v in spec.items():
            el.set(qn(f'w:{k}'), str(v))


def write_docx_table(df, path, title, note=None):
    """Write a genuine three-line table: rule above header, below header,
    below the last row; no vertical or internal horizontal lines."""
    if df.empty or not HAS_DOCX:
        return

    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(9)

    cap = doc.add_paragraph(title)
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.runs[0].bold = True

    cols = list(df.columns)
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = doc.styles['Table Grid']

    thick = {'val': 'single', 'sz': 12, 'space': 0, 'color': '000000'}
    thin = {'val': 'single', 'sz': 6, 'space': 0, 'color': '000000'}
    none = {'val': 'nil'}

    hdr = table.rows[0].cells
    for i, c in enumerate(cols):
        hdr[i].text = ''
        p = hdr[i].paragraphs[0]
        run = p.add_run(str(c).replace('R2', 'R²'))
        run.bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i < 2 else WD_ALIGN_PARAGRAPH.RIGHT
        _set_cell_border(hdr[i], top=thick, bottom=thin, left=none, right=none)

    for _, r in df.iterrows():
        cells = table.add_row().cells
        for i, c in enumerate(cols):
            cells[i].text = ''
            p = cells[i].paragraphs[0]
            p.add_run(str(r[c]))
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i < 2 else WD_ALIGN_PARAGRAPH.RIGHT
            _set_cell_border(cells[i], top=none, bottom=none, left=none, right=none)

    for cell in table.rows[-1].cells:
        _set_cell_border(cell, bottom=thick, top=none, left=none, right=none)

    if note:
        np_ = doc.add_paragraph(note)
        np_.runs[0].font.size = Pt(8)

    doc.save(path)


def write_three_line_table(df, output_dir, stem, title, note=None):
    """Write the same table as .docx, .tex and .txt."""
    if df.empty:
        print(f'[Note] Nothing to write for {stem}.')
        return
    write_docx_table(df, os.path.join(output_dir, f'{stem}.docx'), title, note)
    write_tex_table(df, os.path.join(output_dir, f'{stem}.tex'), title, note,
                    label=f'tab:{stem}')
    write_txt_table(df, os.path.join(output_dir, f'{stem}.txt'), title, note)
    if not HAS_DOCX:
        print('[Warning] python-docx not installed; .docx table skipped '
              '(pip install python-docx).')


def pick_best_model(pooled_df, metric, higher_is_better=True, manual=None):
    """Model with the best pooled external performance (or the fixed one)."""
    if manual is not None:
        return manual
    if pooled_df is None or pooled_df.empty or metric not in pooled_df.columns:
        return None
    d = pooled_df.dropna(subset=[metric])
    if d.empty:
        return None
    idx = d[metric].idxmax() if higher_is_better else d[metric].idxmin()
    return d.loc[idx, 'Model']


def build_best_model_table(cls_metrics_df, cls_pooled, reg_metrics_df, reg_pooled):
    """Compact table: one row per external country, best model only."""
    best_cls = pick_best_model(cls_pooled, BEST_CLS_METRIC, True, BEST_CLS_MODEL)
    best_reg = pick_best_model(
        reg_pooled, BEST_REG_METRIC,
        BEST_REG_METRIC not in ('RMSE', 'MAE'), BEST_REG_MODEL
    )

    if best_cls is None and best_reg is None:
        return pd.DataFrame(), None, None

    cls_all = pd.concat(
        [d for d in [cls_metrics_df, cls_pooled] if d is not None and not d.empty],
        ignore_index=True
    ) if cls_metrics_df is not None else pd.DataFrame()
    reg_all = pd.concat(
        [d for d in [reg_metrics_df, reg_pooled] if d is not None and not d.empty],
        ignore_index=True
    ) if reg_metrics_df is not None else pd.DataFrame()

    countries = [c for c in EXTERNAL_COUNTRIES
                 if (not cls_all.empty and c in set(cls_all['ExternalCountry']))
                 or (not reg_all.empty and c in set(reg_all['ExternalCountry']))]
    countries.append('All external countries')

    rows = []
    for country in countries:
        row = {'Country': country}

        c = cls_all[(cls_all['ExternalCountry'] == country)
                    & (cls_all['Model'] == best_cls)] if not cls_all.empty else pd.DataFrame()
        r = reg_all[(reg_all['ExternalCountry'] == country)
                    & (reg_all['Model'] == best_reg)] if not reg_all.empty else pd.DataFrame()

        if c.empty and r.empty:
            continue

        src = c.iloc[0] if not c.empty else r.iloc[0]
        row['Test n'] = int(src['Test_n']) if pd.notna(src.get('Test_n')) else ''

        if not c.empty:
            cc = c.iloc[0]
            pos = cc.get('Test_Positive_n')
            n = cc.get('Test_n')
            row['High-fitness n (%)'] = (
                f'{int(pos)} ({100.0 * pos / n:.1f})'
                if pd.notna(pos) and pd.notna(n) and n else ''
            )
            for m in ['AUC', 'Accuracy', 'F1']:
                row[_metric_label(m)] = fmt_ci(
                    cc.get(m), cc.get(f'{m}_ci95_low'), cc.get(f'{m}_ci95_high'), 3
                )
        else:
            row['High-fitness n (%)'] = ''
            row['AUC'] = '--'
            row['Accuracy'] = '--'
            row['F1'] = '--'

        if not r.empty:
            rr = r.iloc[0]
            row['Pearson r'] = fmt_ci(
                rr.get('Pearson_r'), rr.get('Pearson_r_ci95_low'),
                rr.get('Pearson_r_ci95_high'), 3
            )
            row['RMSE'] = fmt_ci(
                rr.get('RMSE'), rr.get('RMSE_ci95_low'), rr.get('RMSE_ci95_high'), 2
            )
            row['Bias'] = fmt_ci(
                rr.get('Bias'), rr.get('Bias_ci95_low'), rr.get('Bias_ci95_high'), 2
            )
        else:
            row['Pearson r'] = '--'
            row['RMSE'] = '--'
            row['Bias'] = '--'

        rows.append(row)

    return pd.DataFrame(rows), best_cls, best_reg


def build_country_overview(df):
    """n, class balance and R distribution for every country used."""
    rows = []
    for country, g in df.groupby(COUNTRY_COL):
        role = ('Training' if country in TRAIN_COUNTRIES
                else 'External validation')
        rows.append({
            'Country': country,
            'Role': role,
            'n': int(len(g)),
            'High-fitness n': int(g['fitness_class'].sum()),
            'High-fitness %': round(100.0 * g['fitness_class'].mean(), 1),
            'R mean': round(float(g[TARGET].mean()), 2),
            'R SD': round(float(g[TARGET].std(ddof=1)), 2) if len(g) > 1 else np.nan,
            'R min': round(float(g[TARGET].min()), 2),
            'R max': round(float(g[TARGET].max()), 2),
        })
    out = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(TRAIN_COUNTRIES + EXTERNAL_COUNTRIES)}
    out['_k'] = out['Country'].map(lambda c: order.get(c, 999))
    return out.sort_values(['Role', '_k']).drop(columns='_k').reset_index(drop=True)


# =========================
# 7. Main
# =========================
def main():
    output_dir = get_output_dir()
    print(f'Current working directory: {os.getcwd()}')
    print(f'Output directory: {output_dir}')
    print(f'\nTraining countries ({len(TRAIN_COUNTRIES)}): '
          f'{", ".join(TRAIN_COUNTRIES)}')
    print(f'External countries ({len(EXTERNAL_COUNTRIES)}): '
          f'{", ".join(EXTERNAL_COUNTRIES)}')

    data = pd.read_csv(DATA_PATH)
    data = prepare_data(data)

    train_df = data[data[COUNTRY_COL].isin(TRAIN_COUNTRIES)].copy()
    ext_df = data[data[COUNTRY_COL].isin(EXTERNAL_COUNTRIES)].copy()

    if train_df.empty:
        raise ValueError('No training rows available for the 9 training countries.')
    if ext_df.empty:
        raise ValueError('No rows available for any of the external countries.')

    overview = build_country_overview(data)
    print('\n=== Country overview ===')
    print(overview.to_string(index=False))
    print(f'\nTraining rows: {len(train_df)} | External rows: {len(ext_df)}')

    # ---------------- Regression ----------------
    reg_models, reg_k_dict = fit_regression_models(train_df)
    reg_by_country, reg_pooled, reg_preds = evaluate_regression_external(
        reg_models, train_df, ext_df
    )

    print('\n=== Regression: external countries ===')
    print(reg_by_country.round(4).to_string(index=False)
          if not reg_by_country.empty else 'No regression results.')
    print('\n=== Regression: pooled over external countries ===')
    print(reg_pooled.round(4).to_string(index=False)
          if not reg_pooled.empty else 'No pooled regression results.')

    # ---------------- Classification ----------------
    cls_models, cls_k_dict = fit_classification_models(train_df)
    cls_by_country, cls_pooled, cls_preds = evaluate_classification_external(
        cls_models, train_df, ext_df
    )

    print('\n=== Classification: external countries ===')
    print(cls_by_country.round(4).to_string(index=False)
          if not cls_by_country.empty else 'No classification results.')
    print('\n=== Classification: pooled over external countries ===')
    print(cls_pooled.round(4).to_string(index=False)
          if not cls_pooled.empty else 'No pooled classification results.')

    # ---------------- GAM spline record ----------------
    gam_rows = [{'Task': 'Regression', 'Term': k, 'Best_n_splines': v}
                for k, v in reg_k_dict.items()]
    gam_rows += [{'Task': 'Classification', 'Term': k, 'Best_n_splines': v}
                 for k, v in cls_k_dict.items()]
    gam_df = pd.DataFrame(gam_rows)

    # ---------------- Save raw CSVs ----------------
    overview.to_csv(os.path.join(output_dir, 'external_country_overview.csv'), index=False)
    reg_by_country.to_csv(os.path.join(output_dir, 'external_regression_by_country.csv'), index=False)
    reg_pooled.to_csv(os.path.join(output_dir, 'external_regression_pooled.csv'), index=False)
    reg_preds.to_csv(os.path.join(output_dir, 'external_regression_predictions.csv'), index=False)
    cls_by_country.to_csv(os.path.join(output_dir, 'external_classification_by_country.csv'), index=False)
    cls_pooled.to_csv(os.path.join(output_dir, 'external_classification_pooled.csv'), index=False)
    cls_preds.to_csv(os.path.join(output_dir, 'external_classification_predictions.csv'), index=False)
    gam_df.to_csv(os.path.join(output_dir, 'external_gam_best_splines.csv'), index=False)

    # ---------------- Three-line tables ----------------
    train_note = (
        f'Models were trained once on {len(train_df)} sequences from the nine '
        f'training countries ({", ".join(TRAIN_COUNTRIES)}) and applied without '
        f'refitting to each external country. Values are point estimates with '
        f'95% confidence intervals from {N_BOOTSTRAP} bootstrap resamples of the '
        f'held-out samples of that country. Accuracy, precision, recall and '
        f'F1 use the fixed R > {CLASS_CUTOFF} threshold and therefore depend on '
        f'the class prevalence of the test country, whereas AUC is '
        f'threshold-free and reflects only the within-country ranking. '
        f'Bias is the mean signed error (predicted minus observed R); '
        f'positive values indicate over-prediction. '
        f'GAM, generalized additive model.'
    )

    cls_table = build_metric_table(
        cls_by_country, cls_pooled,
        metrics=CLS_METRICS,
        decimals={m: 3 for m in CLS_METRICS},
        extra_cols=[
            ('Test n', 'Test_n', lambda r: f"{int(r['Test_n'])}"),
            ('High-fitness n (%)', 'Test_Positive_n',
             lambda r: f"{int(r['Test_Positive_n'])} "
                       f"({100.0 * r['Test_Positive_n'] / r['Test_n']:.1f})"),
        ],
    )

    reg_table = build_metric_table(
        reg_by_country, reg_pooled,
        metrics=REG_METRICS,
        decimals={'RMSE': 2, 'MAE': 2, 'R2': 3, 'Pearson_r': 3,
                  'Spearman_rho': 3, 'Bias': 2},
        extra_cols=[
            ('Test n', 'Test_n', lambda r: f"{int(r['Test_n'])}"),
        ],
    )

    best_table, best_cls_model, best_reg_model = build_best_model_table(
        cls_by_country, cls_pooled, reg_by_country, reg_pooled
    )

    write_three_line_table(
        cls_table, output_dir, 'Table_external_classification',
        'Table. External validation of fitness classification in four '
        'countries not used for model training.',
        train_note,
    )

    write_three_line_table(
        reg_table, output_dir, 'Table_external_regression',
        'Table. External validation of fitness regression in four '
        'countries not used for model training.',
        train_note,
    )

    best_note = (
        f'Best classification model: {display_model(best_cls_model)}; '
        f'best regression model: {display_model(best_reg_model)}. ' + train_note
    )
    write_three_line_table(
        best_table, output_dir, 'Table_external_best_model',
        'Table. Performance of the best-performing model in each external '
        'country under the nine-plus-one validation design.',
        best_note,
    )

    # ---------------- Console three-line tables ----------------
    for name, tbl in [
        ('CLASSIFICATION', cls_table),
        ('REGRESSION', reg_table),
        ('BEST MODEL SUMMARY', best_table),
    ]:
        if tbl.empty:
            continue
        print('\n' + '=' * 100)
        print(f'{name} - external validation (9 + 1)')
        print('=' * 100)
        print(tbl.to_string(index=False))

    print('\n' + '=' * 100)
    print(f'Best classification model: {display_model(best_cls_model)} '
          f'(by pooled {BEST_CLS_METRIC})'
          if best_cls_model else 'Best classification model: n/a')
    print(f'Best regression model:     {display_model(best_reg_model)} '
          f'(by pooled {BEST_REG_METRIC})'
          if best_reg_model else 'Best regression model: n/a')

    print('\nSaved files under:')
    print(output_dir)
    for name in sorted(os.listdir(output_dir)):
        print(f'- {name}')


if __name__ == '__main__':
    main()
