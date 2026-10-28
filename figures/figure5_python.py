# -*- coding: utf-8 -*-
"""
Figure 5 in Python
==================

Redraws the country-level leave-one-country-out (LOCO) figure entirely in
Python, replacing the previous R/ggplot2 version, and builds the merged
("mixed-threshold") version of the same figure.

Two figures are produced.

1. ORIGINAL
   Every held-out country is scored with the fixed classification
   boundary R > 60. Read directly from the original LOCO output folder,
   so the numbers are identical to the published Figure 5.

2. MERGED
   The ten countries that were already well predicted keep their original
   R > 60 results, unchanged. Only China, Turkey and Chile are replaced by
   the re-analysis in which the high-fitness class is defined by each of
   those countries' own median R (see loco_relative_fitness.py). The two
   sets of rows are then merged into a single table and the same figure is
   drawn from it.

   Note on the regression panels: shifting the regression target by a
   single constant within a fold leaves RMSE, MAE, R2 and Pearson r
   unchanged, because the residuals are unchanged. The re-analysis
   therefore centres each country on its own median, which is what makes
   the regression panels of the merged figure differ from the original.

Panels (identical layout in both figures)
    A  classification metrics per held-out country, best model
    B  per-country deviation from the mean accuracy
    C  Pearson r and R2 per country, best regression model
    D  RMSE and MAE per country

Inputs
    Any folder holding the standard LOCO output files:
        classification_metrics_by_country_loco.csv
        regression_metrics_by_country_loco.csv

Usage
    python figure5_python.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 0. SETTINGS  (the only block you normally need to edit)
# ============================================================

# folder with the ORIGINAL LOCO results (fixed R > 60 for every country)
DIR_ORIGINAL = "../results/main_analysis"

# folder with the RE-ANALYSIS (high fitness = R > that country's median)
DIR_REDONE = "../results/relative_fitness/loco_relative"

# countries whose rows are taken from DIR_REDONE in the merged figure;
# every other country keeps its DIR_ORIGINAL rows unchanged
REDONE_COUNTRIES = ["China", "Turkey", "Chile"]

OUTPUT_DIR = "../results/figures"

# how to choose the model shown in the figure
#   None -> highest mean Accuracy (classification) / Pearson r (regression)
#   or write a standardized name, e.g. "SVM", "CatBoost"
BEST_CLS_MODEL = None
BEST_REG_MODEL = None

# Panel C: floor for the R2 axis. A single very negative R2 (China is
# -33.8 in the original analysis) otherwise compresses every other bar
# into a flat line. Bars below the floor are truncated and labelled with
# their true value. Set to None to draw the axis untruncated.
R2_FLOOR = -5.0

# Country order on the x axis of both figures.
#   "original" : both figures use the order of the ORIGINAL figure
#                (best to worst accuracy under R > 60), so that the two
#                figures can be compared bar by bar
#   "own"      : each figure is sorted by its own accuracy
#   [ ... ]    : an explicit list of country names
COUNTRY_ORDER = "original"

FIG_SIZE = (15, 10.5)
DPI = 600
SHOW_CHANCE_LINE = True          # dotted line at 0.5 in panel A


# ============================================================
# 1. Model names and colours
# ============================================================

MODEL_DISPLAY = {
    "Stepwise-GAM": "GAM",
    "Stepwise-LogisticGAM": "GAM",
    "RidgeClassifier": "Ridge",
    "SVC": "SVM",
    "SVR": "SVM",
    "RandomForest": "Random Forest",
    "RF": "Random Forest",
}

MODEL_ORDER = ["GAM", "Ridge", "SVM", "Random Forest",
               "AdaBoost", "XGBoost", "LightGBM", "CatBoost"]

COL_CLS = {"Accuracy": "#4C78A8", "Precision": "#72B7B2",
           "Recall": "#E07B39", "F1": "#B07AA1"}
COL_CORR = {"Pearson r": "#E07B39", "R²": "#F2C185"}
COL_ERR = {"RMSE": "#2E8B57", "MAE": "#9BCFB0"}
COL_POS, COL_NEG = "#2166AC", "#B2182B"

CLS_METRICS = ["Accuracy", "Precision", "Recall", "F1"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def display_model(name):
    return MODEL_DISPLAY.get(name, name)


# ============================================================
# 2. Reading and merging LOCO results
# ============================================================

def _detect_country_col(df):
    for c in ("HeldOutCountry", "Country", "country", "Held_out_country"):
        if c in df.columns:
            return c
    raise ValueError(f"No held-out-country column in: {list(df.columns)}")


def read_loco(folder):
    """Read one LOCO output folder -> (classification df, regression df)."""
    out = []
    for fn in ("classification_metrics_by_country_loco.csv",
               "regression_metrics_by_country_loco.csv"):
        path = os.path.join(folder, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df = df.rename(columns={_detect_country_col(df): "Country"})
        df["Model"] = df["Model"].map(display_model).fillna(df["Model"])
        out.append(df)
    return out[0], out[1]


def merge_countries(base, redone, countries):
    """Rows for `countries` come from `redone`; everything else from `base`."""
    missing = sorted(set(countries) - set(redone["Country"]))
    if missing:
        raise ValueError(f"Not present in the re-analysis: {missing}")
    keep = base[~base["Country"].isin(countries)]
    swap = redone[redone["Country"].isin(countries)]
    merged = pd.concat([keep, swap], ignore_index=True)
    # keep only columns shared by both sources, so the merge is consistent
    shared = [c for c in base.columns if c in redone.columns]
    return merged[shared]


def pick_best(df, metric, higher_is_better=True, manual=None):
    if manual is not None:
        return manual
    s = df.groupby("Model")[metric].mean().dropna()
    if s.empty:
        raise ValueError(f"Cannot rank models on '{metric}'.")
    return s.idxmax() if higher_is_better else s.idxmin()


# ============================================================
# 3. Drawing helpers
# ============================================================

def _grouped_bars(ax, categories, series, colors, width=0.80):
    n = len(series)
    x = np.arange(len(categories))
    bw = width / n
    for i, (label, values) in enumerate(series.items()):
        ax.bar(x - width / 2 + bw * (i + 0.5), values, bw,
               label=label, color=colors[label], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    return x


def _panel_tag(ax, letter):
    ax.text(-0.09, 1.14, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")


# ============================================================
# 4. The figure
# ============================================================

def draw_figure(cls_df, reg_df, outfile,
                best_cls=None, best_reg=None,
                r2_floor=R2_FLOOR, note=None, order=None):
    """Draw the four-panel country-level figure and save it.

    `order` fixes the sequence of countries on the x axis. Passing the
    order returned by an earlier call makes two figures directly
    comparable bar by bar. When it is None the countries are sorted by
    the accuracy of this figure's own best model.
    """

    best_cls = pick_best(cls_df, "Accuracy", True, best_cls or BEST_CLS_MODEL)
    best_reg = pick_best(reg_df, "Pearson_r", True, best_reg or BEST_REG_MODEL)

    cc = cls_df[cls_df["Model"] == best_cls].copy()
    rr = reg_df[reg_df["Model"] == best_reg].copy()
    if cc.empty or rr.empty:
        raise ValueError("No rows for the selected model(s).")

    if order is None:
        order = cc.sort_values("Accuracy", ascending=False)["Country"].tolist()
    else:
        order = [c for c in order if c in set(cc["Country"])]
        order += [c for c in cc["Country"] if c not in order]
    cc = cc.set_index("Country").reindex(order)
    rr = rr.set_index("Country").reindex(order)

    mean_acc = cc["Accuracy"].mean()
    mean_rmse = rr["RMSE"].mean()

    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)

    # ---- A: classification metrics -------------------------
    ax = axes[0, 0]
    _grouped_bars(ax, order, {m: cc[m].values for m in CLS_METRICS}, COL_CLS)
    ax.axhline(mean_acc, ls="--", c="0.25", lw=1)
    if SHOW_CHANCE_LINE:
        ax.axhline(0.5, ls=":", c="0.55", lw=1)
    ax.text(len(order) - 0.4, mean_acc, f"mean Accuracy = {mean_acc:.3f}",
            ha="right", va="bottom", fontsize=8.5, color="0.25")
    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Classification performance", fontweight="bold")
    ax.set_title(f"Classification, best model: {best_cls}",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(ncol=4, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.17), fontsize=9)

    # ---- B: deviation from the mean ------------------------
    ax = axes[0, 1]
    delta = cc["Accuracy"].values - mean_acc
    span = max(np.abs(delta).max(), 1e-9)
    for i, v in enumerate(delta):
        ax.bar(i, v, 0.68, color=COL_POS if v >= 0 else COL_NEG,
               alpha=min(1.0, 0.25 + 0.75 * abs(v) / span), edgecolor="none")
        ax.text(i, v, f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, c="0.25", lw=1)
    ax.set_ylim(-span * 1.45, span * 1.45)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Accuracy − mean across countries", fontweight="bold")
    ax.set_title(f"Per-country deviation ({best_cls})",
                 fontsize=11, fontweight="bold", loc="left")

    # ---- C: Pearson r and R2 -------------------------------
    ax = axes[1, 0]
    r2 = rr["R2"].values.astype(float)
    r2_plot = r2.copy()
    truncated = np.zeros(len(r2), dtype=bool)
    if r2_floor is not None:
        truncated = r2 < r2_floor
        r2_plot = np.clip(r2, r2_floor, None)
    _grouped_bars(ax, order,
                  {"Pearson r": rr["Pearson_r"].values, "R²": r2_plot},
                  COL_CORR, width=0.78)
    ax.axhline(0, c="0.25", lw=0.9)
    lo = min(0.0, float(np.nanmin(r2_plot))) * 1.15
    ax.set_ylim(lo if lo < 0 else -0.05, 1.08)
    for i, (orig, tr) in enumerate(zip(r2, truncated)):
        if tr:
            ax.text(i + 0.195, lo * 0.97, f"{orig:.1f}", ha="center",
                    va="bottom", fontsize=7.5, rotation=90, color="#7a1020")
    ax.set_ylabel("Predicted vs observed fitness", fontweight="bold")
    ttl = f"Regression, best model: {best_reg}"
    if truncated.any():
        ttl += f"  (R² below {r2_floor:g} truncated and labelled)"
    ax.set_title(ttl, fontsize=11, fontweight="bold", loc="left")
    ax.legend(ncol=2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), fontsize=9)

    # ---- D: prediction error -------------------------------
    ax = axes[1, 1]
    _grouped_bars(ax, order,
                  {"RMSE": rr["RMSE"].values, "MAE": rr["MAE"].values},
                  COL_ERR, width=0.78)
    ax.axhline(mean_rmse, ls="--", c="0.25", lw=1)
    ax.text(len(order) - 0.4, mean_rmse, f"mean RMSE = {mean_rmse:.2f}",
            ha="right", va="bottom", fontsize=8.5, color="0.25")
    ax.set_ylabel("Prediction error (lower is better)", fontweight="bold")
    ax.set_title(f"Prediction error, {best_reg}",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(ncol=2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), fontsize=9)

    for letter, ax in zip("ABCD", axes.ravel()):
        _panel_tag(ax, letter)

    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=8.5, color="0.3")

    fig.tight_layout(rect=[0, 0.02 if note else 0, 1, 0.99])
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    fig.savefig(outfile.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {outfile}   [cls={best_cls}, reg={best_reg}]")
    return best_cls, best_reg, order


# ============================================================
# 5. Main
# ============================================================

def main():
    cls_o, reg_o = read_loco(DIR_ORIGINAL)
    cls_r, reg_r = read_loco(DIR_REDONE)

    # ---- figure 1: original, fixed R > 60 everywhere -------
    fixed = None if COUNTRY_ORDER == "own" else (
        None if COUNTRY_ORDER == "original" else list(COUNTRY_ORDER))
    _, _, order_original = draw_figure(
        cls_o, reg_o,
        os.path.join(OUTPUT_DIR, "Figure5_original_R60.png"),
        order=fixed,
        note="All held-out countries scored with the fixed "
             "classification boundary R > 60.")

    # the merged figure reuses this order unless told otherwise
    if COUNTRY_ORDER == "own":
        shared_order = None
    elif COUNTRY_ORDER == "original":
        shared_order = order_original
    else:
        shared_order = list(COUNTRY_ORDER)

    # ---- figure 2: merged ----------------------------------
    cls_m = merge_countries(cls_o, cls_r, REDONE_COUNTRIES)
    reg_m = merge_countries(reg_o, reg_r, REDONE_COUNTRIES)

    swapped = ", ".join(REDONE_COUNTRIES)
    draw_figure(cls_m, reg_m,
                os.path.join(OUTPUT_DIR, "Figure5_merged_median_threshold.png"),
                order=shared_order,
                note=f"{swapped} scored against their own median R; all other "
                     f"countries keep the original R > 60 results. Countries "
                     f"appear in the same order as in the preceding figure.")

    # the merged tables themselves, so the figure can be audited
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cls_m.assign(Threshold=np.where(cls_m.Country.isin(REDONE_COUNTRIES),
                                    "country median R", "R > 60")) \
         .sort_values(["Model", "Country"]) \
         .to_csv(os.path.join(OUTPUT_DIR,
                 "merged_classification_metrics_by_country.csv"), index=False)
    reg_m.assign(Threshold=np.where(reg_m.Country.isin(REDONE_COUNTRIES),
                                    "country median R", "R > 60")) \
         .sort_values(["Model", "Country"]) \
         .to_csv(os.path.join(OUTPUT_DIR,
                 "merged_regression_metrics_by_country.csv"), index=False)

    # ---- a small comparison table --------------------------
    rows = []
    for c in REDONE_COUNTRIES:
        for model in sorted(set(cls_o["Model"]) & set(cls_r["Model"])):
            a = cls_o[(cls_o.Country == c) & (cls_o.Model == model)]
            b = cls_r[(cls_r.Country == c) & (cls_r.Model == model)]
            if a.empty or b.empty:
                continue
            rows.append(dict(Country=c, Model=model,
                             Accuracy_R60=float(a.Accuracy.iloc[0]),
                             Accuracy_median=float(b.Accuracy.iloc[0])))
    comp = pd.DataFrame(rows)
    comp["Change"] = comp.Accuracy_median - comp.Accuracy_R60
    out_csv = os.path.join(OUTPUT_DIR, "swapped_countries_comparison.csv")
    comp.to_csv(out_csv, index=False)
    print("\n" + comp.round(3).to_string(index=False))
    print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    main()
