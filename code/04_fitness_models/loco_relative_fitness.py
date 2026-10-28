# -*- coding: utf-8 -*-
"""
LOCO with an absolute versus a within-country (relative) definition of fitness
=============================================================================

Diagnosis being tested
----------------------
Under the original leave-one-country-out (LOCO) analysis, three held-out
countries performed far worse than the rest (China, Turkey, Chile). The
cause is a country-level shift in R that the four viral-trait predictors
do not explain: China's lineages sit about 40 units of R below the
training countries and Turkey's about 24 above, while Chile's R
distribution straddles the fixed R > 60 class boundary almost exactly.

Two outcome definitions are therefore compared, using identical data,
folds and models:

  absolute  (the original analysis)
      classification : high fitness = R > CLASS_CUTOFF (60)
      regression     : target = R

  relative  (this sensitivity analysis)
      classification : high fitness = R > that country's own median R
      regression     : target = R - that country's own median R

Both the training countries and the held-out country are transformed the
same way, so the model learns, and is evaluated on, the position of a
lineage relative to the fitness level of its own country.

Important caveat
----------------
A country's median R is estimated from that country's own data, so the
relative analysis assumes the overall fitness level of the target country
is known (as it would be from routine surveillance). It is therefore not
a fully blind prediction: it isolates how much of the LOCO failure comes
from the unknown country-level level, as opposed to an inability to rank
lineages within a country.

Outputs (per mode, in <OUT>/loco_<mode>/) use the same file names and
column layout as the original LOCO outputs, so the existing Figure 5
R script can read them unchanged.
"""

import os
import sys
import warnings
import operator
from functools import reduce

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pygam import LinearGAM, LogisticGAM, s, te

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
)
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


# ---------------- settings ----------------
DATA_PATH = os.environ.get(
    "FITNESS_DATA",
    "../../data/processed/merged_binding_final_updated.csv",
)
OUT_BASE = os.environ.get("FITNESS_OUT", "../../results/relative_fitness")

SEED = 42
R_MIN, R_MAX = 10, 200
CLASS_CUTOFF = 60

FEATURES = ['pred_log10Kd_ACE2', 'binding_retained', 'spike_entry_score', 'SDI']
TARGET, COUNTRY_COL = 'R', 'country'

# the 13 countries of the main LOCO analysis
COUNTRIES = ['Bangladesh', 'Brazil', 'Cambodia', 'Chile', 'China', 'Germany',
             'India', 'Mexico', 'Poland', 'Senegal', 'Turkey', 'UK', 'USA']


def load_data():
    df = pd.read_csv(DATA_PATH)
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].replace([np.inf, -np.inf], np.nan)
    df = df[(df[TARGET] >= R_MIN) & (df[TARGET] <= R_MAX)]
    df = df[df[COUNTRY_COL].isin(COUNTRIES)]
    df = df.dropna(subset=FEATURES + [TARGET, COUNTRY_COL]).copy()

    # per-country median R, used by the relative mode
    med = df.groupby(COUNTRY_COL)[TARGET].median()
    df['country_median_R'] = df[COUNTRY_COL].map(med)
    return df, med


def apply_mode(df, mode):
    """Add the outcome columns for the requested mode."""
    df = df.copy()
    if mode == 'absolute':
        df['y_reg'] = df[TARGET]
        df['y_cls'] = (df[TARGET] > CLASS_CUTOFF).astype(int)
    elif mode == 'relative':
        df['y_reg'] = df[TARGET] - df['country_median_R']
        df['y_cls'] = (df[TARGET] > df['country_median_R']).astype(int)
    else:
        raise ValueError(mode)
    return df


# ---------------- metrics ----------------
def safe_pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan
    return float(pearsonr(a, b)[0])


def reg_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y, p))),
        'MAE': float(mean_absolute_error(y, p)),
        'R2': float(r2_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        'Pearson_r': safe_pearson(y, p),
    }


def cls_metrics(y, p, proba=None):
    out = {
        'Accuracy': float(accuracy_score(y, p)),
        'Precision': float(precision_score(y, p, zero_division=0)),
        'Recall': float(recall_score(y, p, zero_division=0)),
        'F1': float(f1_score(y, p, zero_division=0)),
        'AUC': np.nan,
    }
    if proba is not None and len(np.unique(y)) == 2 and np.isfinite(proba).all():
        try:
            out['AUC'] = float(roc_auc_score(y, proba))
        except Exception:
            pass
    return out


# ---------------- GAM ----------------
def choose_best_terms(X, y, names, gam_class, k_range=range(4, 9)):
    main, inter, kd = [], [], {}
    sdi = names.index('SDI')
    for i, nm in enumerate(names):
        bk, ba = None, np.inf
        for k in k_range:
            try:
                a = gam_class(s(i, n_splines=k)).fit(X, y).statistics_['AIC']
                if a < ba:
                    ba, bk = a, k
            except Exception:
                continue
        if bk:
            main.append(s(i, n_splines=bk)); kd[nm] = bk
    for i, nm in enumerate(names):
        if nm == 'SDI':
            continue
        bk, ba = None, np.inf
        for k in k_range:
            try:
                a = gam_class(te(i, sdi, n_splines=k)).fit(X, y).statistics_['AIC']
                if a < ba:
                    ba, bk = a, k
            except Exception:
                continue
        if bk:
            inter.append(te(i, sdi, n_splines=bk)); kd[f'{nm}:SDI'] = bk
    return main + inter, kd


def stepwise_gam(X, y, terms, gam_class, thr=0.05):
    if not terms:
        return None
    tw = list(terms)
    while tw:
        try:
            g = gam_class(reduce(operator.add, tw)).fit(X, y)
        except Exception:
            return None
        pv = np.array(g.statistics_['p_values'][:-1])
        if len(pv) == 0 or np.all(pv < thr):
            break
        i = int(np.argmax(pv))
        if pv[i] < thr:
            break
        del tw[i]
    if not tw:
        return None
    try:
        return gam_class(reduce(operator.add, tw)).fit(X, y)
    except Exception:
        return None


# ---------------- model zoo ----------------
def reg_models():
    return [
        ('XGBoost', XGBRegressor(random_state=SEED, n_estimators=300, max_depth=4,
                                 learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                                 tree_method='hist', device='cpu', n_jobs=4)),
        ('Random Forest', RandomForestRegressor(random_state=SEED, n_estimators=300,
                                                min_samples_leaf=2, n_jobs=4)),
        ('Ridge', make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ('SVR', make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1, kernel='rbf'))),
        ('AdaBoost', AdaBoostRegressor(random_state=SEED, n_estimators=300, learning_rate=0.05)),
        ('LightGBM', LGBMRegressor(random_state=SEED, n_estimators=300, learning_rate=0.05,
                                   num_leaves=31, subsample=0.9, colsample_bytree=0.9, verbose=-1)),
        ('CatBoost', CatBoostRegressor(random_state=SEED, iterations=300, depth=6,
                                       learning_rate=0.05, loss_function='RMSE', verbose=False)),
    ]


def cls_models():
    return [
        ('XGBoost', XGBClassifier(random_state=SEED, n_estimators=300, max_depth=4,
                                  learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                                  eval_metric='logloss', tree_method='hist', device='cpu', n_jobs=4)),
        ('Random Forest', RandomForestClassifier(random_state=SEED, n_estimators=300,
                                                 min_samples_leaf=2, n_jobs=4)),
        ('RidgeClassifier', make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))),
        ('SVC', make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf'))),
        ('AdaBoost', AdaBoostClassifier(random_state=SEED, n_estimators=300, learning_rate=0.05)),
        ('LightGBM', LGBMClassifier(random_state=SEED, n_estimators=300, learning_rate=0.05,
                                    num_leaves=31, subsample=0.9, colsample_bytree=0.9, verbose=-1)),
        ('CatBoost', CatBoostClassifier(random_state=SEED, iterations=300, depth=6,
                                        learning_rate=0.05, loss_function='Logloss', verbose=False)),
    ]


def predict_binary(name, model, X):
    if name == 'Stepwise-LogisticGAM':
        pr = np.asarray(model.predict_proba(X)).ravel()
        return (pr >= 0.5).astype(int), pr
    pred = np.asarray(model.predict(X)).ravel().astype(int)
    if hasattr(model, 'predict_proba'):
        try:
            pr = np.asarray(model.predict_proba(X))[:, 1]
        except Exception:
            pr = np.full(len(pred), np.nan)
    elif hasattr(model, 'decision_function'):
        raw = np.asarray(model.decision_function(X)).ravel()
        pr = 1 / (1 + np.exp(-raw))
    else:
        pr = np.full(len(pred), np.nan)
    return pred, pr


# ---------------- LOCO ----------------
def run_loco(df, mode):
    reg_rows, cls_rows, reg_pred, cls_pred = [], [], [], []
    countries = sorted(df[COUNTRY_COL].unique())

    for held in countries:
        tr = df[df[COUNTRY_COL] != held]
        va = df[df[COUNTRY_COL] == held]
        Xtr, Xva = tr[FEATURES].values, va[FEATURES].values
        print(f'[{mode}] held-out: {held}  train={len(tr)} valid={len(va)}', flush=True)

        # ---- regression ----
        ytr, yva = tr['y_reg'].values, va['y_reg'].values
        terms, _ = choose_best_terms(Xtr, ytr, FEATURES, LinearGAM)
        gam = stepwise_gam(Xtr, ytr, terms, LinearGAM)
        models = ([('Stepwise-GAM', gam)] if gam is not None else []) + reg_models()
        for nm, m in models:
            try:
                if nm != 'Stepwise-GAM':
                    m.fit(Xtr, ytr)
                p = np.asarray(m.predict(Xva)).ravel()
            except Exception as e:
                print(f'   [warn] reg {nm}: {e}'); continue
            reg_rows.append(dict(HeldOutCountry=held, Model=nm, Train_n=len(tr),
                                 Valid_n=len(va), **reg_metrics(yva, p)))
            reg_pred.append(pd.DataFrame(dict(HeldOutCountry=held, Model=nm,
                                              Observed=yva, Predicted=p)))

        # ---- classification ----
        ytr_c, yva_c = tr['y_cls'].values, va['y_cls'].values
        if len(np.unique(ytr_c)) < 2:
            continue
        terms, _ = choose_best_terms(Xtr, ytr_c, FEATURES, LogisticGAM)
        gamc = stepwise_gam(Xtr, ytr_c, terms, LogisticGAM)
        models = ([('Stepwise-LogisticGAM', gamc)] if gamc is not None else []) + cls_models()
        for nm, m in models:
            try:
                if nm != 'Stepwise-LogisticGAM':
                    m.fit(Xtr, ytr_c)
                pred, proba = predict_binary(nm, m, Xva)
            except Exception as e:
                print(f'   [warn] cls {nm}: {e}'); continue
            cls_rows.append(dict(HeldOutCountry=held, Model=nm, Train_n=len(tr),
                                 Valid_n=len(va), Train_Positive_n=int(ytr_c.sum()),
                                 Valid_Positive_n=int(yva_c.sum()),
                                 **cls_metrics(yva_c, pred, proba)))
            cls_pred.append(pd.DataFrame(dict(HeldOutCountry=held, Model=nm,
                                              Observed_Class=yva_c, Predicted_Class=pred,
                                              Predicted_Prob=proba)))

    return (pd.DataFrame(reg_rows), pd.DataFrame(cls_rows),
            pd.concat(reg_pred, ignore_index=True) if reg_pred else pd.DataFrame(),
            pd.concat(cls_pred, ignore_index=True) if cls_pred else pd.DataFrame())


def main():
    df, med = load_data()
    print('per-country median R:'); print(med.round(2).to_string())

    for mode in ['absolute', 'relative']:
        out = os.path.join(OUT_BASE, f'loco_{mode}')
        os.makedirs(out, exist_ok=True)
        d = apply_mode(df, mode)
        reg, cls, rp, cp = run_loco(d, mode)
        reg.to_csv(os.path.join(out, 'regression_metrics_by_country_loco.csv'), index=False)
        cls.to_csv(os.path.join(out, 'classification_metrics_by_country_loco.csv'), index=False)
        rp.to_csv(os.path.join(out, 'regression_predictions_loco.csv'), index=False)
        cp.to_csv(os.path.join(out, 'classification_predictions_loco.csv'), index=False)
        print(f'[{mode}] saved to {out}', flush=True)

    med.rename('median_R').to_frame().to_csv(
        os.path.join(OUT_BASE, 'country_median_R.csv'))
    print('DONE')


if __name__ == '__main__':
    main()
