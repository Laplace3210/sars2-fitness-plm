# ============================================================
# SARS-CoV-2 fitness prediction
# FIGURE 5 (REVISED) :
#   Country-level performance of the best LOCO model
#
# Revision note
# ------------------------------------------------------------
# Reviewer comment:
#   "Here it would be interesting to see a bar chart for each
#    country showing performance for the best model; maybe there
#    are some especially good or bad countries in terms of
#    generalisability"
#
# Figure 5 now shows, for the best-performing model, the
# prediction performance in EACH held-out country, instead of
# only the mean across countries.
#
# The previous Figure 5 (model-level means + sensitivity
# heatmaps) is NOT deleted: it is still produced and saved,
# but as a Supplementary Figure.
#
# Main analysis:
#   loco_results_sens1
#
# Sensitivity analyses:
#   loco_results
#   loco_results_sens2
#   loco_results_sens3
# ============================================================


library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(writexl)


# ============================================================
# 0. USER SETTINGS  (the only block you normally need to edit)
# ============================================================

# ---- Which model is "the best model" shown in Figure 5? -----
#
# NA  -> chosen automatically from the data (see below)
# or  -> write the name yourself, e.g. "SVM", "AdaBoost"
#
# Model names must be the STANDARDIZED names, i.e. one of:
#   "GAM", "Ridge", "SVM", "Random Forest",
#   "AdaBoost", "XGBoost", "LightGBM", "CatBoost"

best_cls_model_manual <- NA_character_   # e.g. "SVM"
best_reg_model_manual <- NA_character_   # e.g. "AdaBoost"


# ---- Criteria used when the choice is automatic -------------

cls_select_metric <- "Accuracy"      # "Accuracy", "F1", "Precision", "Recall"
reg_select_metric <- "Pearson_r"     # "Pearson_r", "R2", "RMSE", "MAE"


# ---- Use ONE model for both tasks? --------------------------
#
# TRUE  : the classification winner is also used for the
#         regression panels (simpler message: "the best model")
# FALSE : classification and regression winners are chosen
#         independently (each panel is labelled with its model)

force_same_model <- FALSE


# ---- Country ordering in all panels ------------------------
#
# Countries are sorted (best -> worst) by this metric of the
# best CLASSIFICATION model, and the SAME order is used in
# every panel so that columns can be compared vertically.
#
# Use "alphabetical" to keep countries in alphabetical order.

country_order_by <- "Accuracy"       # or "alphabetical"


# ---- Cosmetic options ---------------------------------------

label_panel_values <- FALSE   # TRUE = print numbers on panel A bars
show_chance_line   <- TRUE    # TRUE = dashed line at 0.5 in panel A
figure5_width      <- 14
figure5_height      <- 10


# ============================================================
# 1. Paths
# ============================================================

# Folder names follow the labels used in the Supplementary Methods:
#   main_analysis = 13 countries (the nine study countries + Chile,
#   Poland, Mexico and China); sensitivity1 = 12 countries (without
#   China); sensitivity2 = 14 (+Denmark, Japan); sensitivity3 = 13 (+Nepal).
base_dir <- "../results"

output_dir <- file.path(
  base_dir,
  "paper_figures"
)

if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}


scenario_info <- tibble(
  Scenario = c(
    "Main analysis",
    "Sensitivity 1",
    "Sensitivity 2",
    "Sensitivity 3"
  ),
  
  Folder = c(
    "main_analysis",
    "sensitivity1",
    "sensitivity2",
    "sensitivity3"
  )
)


main_folder <- file.path(
  base_dir,
  scenario_info$Folder[scenario_info$Scenario == "Main analysis"]
)


# ============================================================
# 2. Standardize model names
# ============================================================

standardize_model <- function(x) {
  
  recode(
    x,
    
    # GAM
    "Stepwise-LogisticGAM" = "GAM",
    "Stepwise-GAM" = "GAM",
    
    # Ridge
    "RidgeClassifier" = "Ridge",
    
    # SVM
    "SVC" = "SVM",
    "SVR" = "SVM",
    
    # Random forest (common spellings)
    "RandomForest" = "Random Forest",
    "RandomForestClassifier" = "Random Forest",
    "RandomForestRegressor" = "Random Forest",
    "RF" = "Random Forest",
    
    # Boosting (common spellings)
    "AdaBoostClassifier" = "AdaBoost",
    "AdaBoostRegressor" = "AdaBoost",
    "XGB" = "XGBoost",
    "LGBM" = "LightGBM",
    "LGBMClassifier" = "LightGBM",
    "LGBMRegressor" = "LightGBM"
  )
}


model_order <- c(
  "GAM",
  "Ridge",
  "SVM",
  "Random Forest",
  "AdaBoost",
  "XGBoost",
  "LightGBM",
  "CatBoost"
)


scenario_order <- c(
  "Main analysis",
  "Sensitivity 1",
  "Sensitivity 2",
  "Sensitivity 3"
)


# Warn (instead of silently producing NA) if a model name in the
# CSV files is not covered by model_order
check_model_names <- function(x, where) {
  
  unknown <- setdiff(
    unique(as.character(x)),
    model_order
  )
  
  if (length(unknown) > 0) {
    warning(
      "Unrecognised model name(s) in ", where, ": ",
      paste(unknown, collapse = ", "),
      "\n  -> add them to standardize_model() / model_order,",
      " otherwise they become NA and disappear from the figures.",
      call. = FALSE
    )
  }
  
  invisible(unknown)
}


# ============================================================
# 3. Helper functions
# ============================================================

read_cls_summary <- function(folder, scenario) {
  
  file_path <- file.path(
    base_dir,
    folder,
    "classification_metrics_summary_loco.csv"
  )
  
  df <- read_csv(
    file_path,
    show_col_types = FALSE
  )
  
  df %>%
    mutate(
      Scenario = scenario,
      Model = standardize_model(Model)
    )
}


read_reg_summary <- function(folder, scenario) {
  
  file_path <- file.path(
    base_dir,
    folder,
    "regression_metrics_summary_loco.csv"
  )
  
  df <- read_csv(
    file_path,
    show_col_types = FALSE
  )
  
  df %>%
    mutate(
      Scenario = scenario,
      Model = standardize_model(Model)
    )
}


read_cls_overall <- function(folder, scenario) {
  
  file_path <- file.path(
    base_dir,
    folder,
    "classification_metrics_overall_loco.csv"
  )
  
  df <- read_csv(
    file_path,
    show_col_types = FALSE
  )
  
  df %>%
    mutate(
      Scenario = scenario,
      Model = standardize_model(Model)
    )
}


read_reg_overall <- function(folder, scenario) {
  
  file_path <- file.path(
    base_dir,
    folder,
    "regression_metrics_overall_loco.csv"
  )
  
  df <- read_csv(
    file_path,
    show_col_types = FALSE
  )
  
  df %>%
    mutate(
      Scenario = scenario,
      Model = standardize_model(Model)
    )
}


# ------------------------------------------------------------
# NEW: country-level readers
# ------------------------------------------------------------

# The name of the held-out-country column is detected, so the
# script keeps working if it is called "Country" instead of
# "HeldOutCountry".

country_col_candidates <- c(
  "HeldOutCountry",
  "Held_out_country",
  "HeldOut_Country",
  "HeldOut",
  "Test_country",
  "TestCountry",
  "Country",
  "country"
)

# Optional column with the number of test sequences / lineages
# in the held-out country (used only for annotation, if present)
size_col_candidates <- c(
  "N_test",
  "n_test",
  "N",
  "n",
  "N_samples",
  "n_samples",
  "Support",
  "N_sequences"
)


detect_col <- function(df, candidates) {
  
  hit <- intersect(candidates, names(df))
  
  if (length(hit) == 0) {
    return(NA_character_)
  }
  
  hit[1]
}


read_country_file <- function(folder, scenario, file_name) {
  
  file_path <- file.path(
    base_dir,
    folder,
    file_name
  )
  
  if (!file.exists(file_path)) {
    
    message(
      "Skipped (file not found): ",
      file_path
    )
    
    return(NULL)
  }
  
  df <- read_csv(
    file_path,
    show_col_types = FALSE
  )
  
  country_col <- detect_col(df, country_col_candidates)
  
  if (is.na(country_col)) {
    stop(
      "No held-out-country column found in:\n  ",
      file_path,
      "\n  Columns present: ",
      paste(names(df), collapse = ", "),
      "\n  -> add the correct name to 'country_col_candidates'."
    )
  }
  
  size_col <- detect_col(df, size_col_candidates)
  
  # rename the held-out-country column to "Country"
  # (avoid creating duplicated names if "Country" already exists)
  if (country_col != "Country") {
    
    if ("Country" %in% names(df)) {
      names(df)[names(df) == "Country"] <- "Country_original"
    }
    
    names(df)[names(df) == country_col] <- "Country"
  }
  
  df <- df %>%
    mutate(
      Scenario = scenario,
      Model = standardize_model(Model),
      Country = as.character(Country)
    )
  
  if (!is.na(size_col)) {
    df <- df %>%
      mutate(N_test = suppressWarnings(as.numeric(.data[[size_col]])))
  } else {
    df <- df %>%
      mutate(N_test = NA_real_)
  }
  
  df
}


read_cls_country <- function(folder, scenario) {
  
  read_country_file(
    folder,
    scenario,
    "classification_metrics_by_country_loco.csv"
  )
}


read_reg_country <- function(folder, scenario) {
  
  read_country_file(
    folder,
    scenario,
    "regression_metrics_by_country_loco.csv"
  )
}


# ------------------------------------------------------------
# Long format with mean (+ 95% CI if several folds per country)
# ------------------------------------------------------------
#
# In a standard LOCO design there is exactly ONE fold per
# (Model, Country), so no error bars are drawn.
# If the file contains repeated runs / seeds per country, the
# mean and a 95% CI are computed automatically and error bars
# appear.

country_to_long <- function(df, metrics) {
  
  metrics <- intersect(metrics, names(df))
  
  if (length(metrics) == 0) {
    stop(
      "None of the expected metric columns were found. ",
      "Columns present: ",
      paste(names(df), collapse = ", ")
    )
  }
  
  df %>%
    select(
      Model,
      Country,
      Scenario,
      N_test,
      all_of(metrics)
    ) %>%
    
    # guard against metric columns read as text (e.g. "NA", "inf")
    mutate(
      across(
        all_of(metrics),
        ~ suppressWarnings(as.numeric(.x))
      )
    ) %>%
    
    pivot_longer(
      cols = all_of(metrics),
      names_to = "Metric",
      values_to = "Value"
    ) %>%
    
    group_by(Model, Country, Scenario, Metric) %>%
    
    summarise(
      Mean = mean(Value, na.rm = TRUE),
      SD = sd(Value, na.rm = TRUE),
      N_folds = sum(!is.na(Value)),
      N_test = suppressWarnings(max(N_test, na.rm = TRUE)),
      .groups = "drop"
    ) %>%
    
    mutate(
      N_test = ifelse(is.finite(N_test), N_test, NA_real_),
      SE = ifelse(N_folds > 1, SD / sqrt(N_folds), NA_real_),
      CI_low = ifelse(!is.na(SE), Mean - 1.96 * SE, NA_real_),
      CI_high = ifelse(!is.na(SE), Mean + 1.96 * SE, NA_real_)
    )
}


# ------------------------------------------------------------
# Pick the best model from the country-level results
# ------------------------------------------------------------

pick_best_model <- function(long_df,
                            metric,
                            higher_better = TRUE) {
  
  ranking <- long_df %>%
    filter(Metric == metric) %>%
    group_by(Model) %>%
    summarise(
      Score = mean(Mean, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    filter(!is.na(Model), !is.na(Score))
  
  ranking <- if (higher_better) {
    arrange(ranking, desc(Score))
  } else {
    arrange(ranking, Score)
  }
  
  if (nrow(ranking) == 0) {
    stop(
      "Could not rank models on metric '", metric, "'."
    )
  }
  
  list(
    best = as.character(ranking$Model[1]),
    ranking = ranking
  )
}


metric_higher_is_better <- function(metric) {
  
  !(metric %in% c("RMSE", "MAE"))
}


# ============================================================
# 4. Read all scenarios
# ============================================================

cls_summary_all <- bind_rows(
  
  lapply(
    seq_len(nrow(scenario_info)),
    
    function(i) {
      
      read_cls_summary(
        scenario_info$Folder[i],
        scenario_info$Scenario[i]
      )
      
    }
  )
)


reg_summary_all <- bind_rows(
  
  lapply(
    seq_len(nrow(scenario_info)),
    
    function(i) {
      
      read_reg_summary(
        scenario_info$Folder[i],
        scenario_info$Scenario[i]
      )
      
    }
  )
)


cls_overall_all <- bind_rows(
  
  lapply(
    seq_len(nrow(scenario_info)),
    
    function(i) {
      
      read_cls_overall(
        scenario_info$Folder[i],
        scenario_info$Scenario[i]
      )
      
    }
  )
)


reg_overall_all <- bind_rows(
  
  lapply(
    seq_len(nrow(scenario_info)),
    
    function(i) {
      
      read_reg_overall(
        scenario_info$Folder[i],
        scenario_info$Scenario[i]
      )
      
    }
  )
)


# ---- country-level results, all scenarios -------------------

cls_country_all <- bind_rows(
  
  lapply(
    seq_len(nrow(scenario_info)),
    
    function(i) {
      
      read_cls_country(
        scenario_info$Folder[i],
        scenario_info$Scenario[i]
      )
      
    }
  )
)


reg_country_all <- bind_rows(
  
  lapply(
    seq_len(nrow(scenario_info)),
    
    function(i) {
      
      read_reg_country(
        scenario_info$Folder[i],
        scenario_info$Scenario[i]
      )
      
    }
  )
)


if (nrow(cls_country_all) == 0) {
  stop(
    "No country-level classification results were read.\n",
    "  Expected: <folder>/classification_metrics_by_country_loco.csv\n",
    "  under: ", base_dir
  )
}

if (nrow(reg_country_all) == 0) {
  stop(
    "No country-level regression results were read.\n",
    "  Expected: <folder>/regression_metrics_by_country_loco.csv\n",
    "  under: ", base_dir
  )
}


check_model_names(cls_country_all$Model, "classification_metrics_by_country_loco.csv")
check_model_names(reg_country_all$Model, "regression_metrics_by_country_loco.csv")


# ============================================================
# 5. Factor ordering
# ============================================================

cls_summary_all <- cls_summary_all %>%
  mutate(
    Model = factor(Model, levels = model_order),
    Scenario = factor(
      Scenario,
      levels = scenario_order
    )
  )


reg_summary_all <- reg_summary_all %>%
  mutate(
    Model = factor(Model, levels = model_order),
    Scenario = factor(
      Scenario,
      levels = scenario_order
    )
  )


cls_overall_all <- cls_overall_all %>%
  mutate(
    Model = factor(Model, levels = model_order),
    Scenario = factor(
      Scenario,
      levels = scenario_order
    )
  )


reg_overall_all <- reg_overall_all %>%
  mutate(
    Model = factor(Model, levels = model_order),
    Scenario = factor(
      Scenario,
      levels = scenario_order
    )
  )


cls_country_all <- cls_country_all %>%
  mutate(
    Model = factor(Model, levels = model_order),
    Scenario = factor(Scenario, levels = scenario_order)
  )


reg_country_all <- reg_country_all %>%
  mutate(
    Model = factor(Model, levels = model_order),
    Scenario = factor(Scenario, levels = scenario_order)
  )


# ============================================================
# 6. COUNTRY-LEVEL DATA FOR THE MAIN ANALYSIS
# ============================================================

cls_metrics <- c("Accuracy", "Precision", "Recall", "F1")
reg_metrics <- c("Pearson_r", "R2", "RMSE", "MAE")


# the metric actually used for sorting / reference lines
# ("alphabetical" only changes the ORDER, not the metric shown)
order_metric <- if (country_order_by %in% cls_metrics) {
  country_order_by
} else {
  "Accuracy"
}


cls_country_long <- cls_country_all %>%
  filter(Scenario == "Main analysis") %>%
  country_to_long(cls_metrics)


reg_country_long <- reg_country_all %>%
  filter(Scenario == "Main analysis") %>%
  country_to_long(reg_metrics)


# ------------------------------------------------------------
# 6.1 Best model
# ------------------------------------------------------------

cls_rank <- pick_best_model(
  cls_country_long,
  cls_select_metric,
  metric_higher_is_better(cls_select_metric)
)


reg_rank <- pick_best_model(
  reg_country_long,
  reg_select_metric,
  metric_higher_is_better(reg_select_metric)
)


best_cls_model <- if (!is.na(best_cls_model_manual)) {
  best_cls_model_manual
} else {
  cls_rank$best
}


best_reg_model <- if (!is.na(best_reg_model_manual)) {
  best_reg_model_manual
} else if (force_same_model) {
  best_cls_model
} else {
  reg_rank$best
}


cat("\n------------------------------------------------------------\n")
cat("Model ranking across held-out countries (main analysis)\n")
cat("------------------------------------------------------------\n")

cat("\nClassification, mean", cls_select_metric, ":\n")
print(as.data.frame(cls_rank$ranking), row.names = FALSE)

cat("\nRegression, mean", reg_select_metric, ":\n")
print(as.data.frame(reg_rank$ranking), row.names = FALSE)

cat(
  "\nModel used in Figure 5 (classification): ",
  best_cls_model,
  "\nModel used in Figure 5 (regression):     ",
  best_reg_model,
  "\n------------------------------------------------------------\n\n"
)


# ------------------------------------------------------------
# 6.2 Country order (same in every panel)
# ------------------------------------------------------------

# every country appearing in either task
all_countries <- sort(
  union(
    unique(cls_country_long$Country),
    unique(reg_country_long$Country)
  )
)


order_ref <- cls_country_long %>%
  filter(
    Model == best_cls_model,
    Metric == country_order_by
  )


if (country_order_by == "alphabetical" || nrow(order_ref) == 0) {
  
  country_levels <- all_countries
  
} else {
  
  country_levels <- order_ref %>%
    arrange(desc(Mean)) %>%
    pull(Country)
  
  # keep any country that is missing from the reference metric
  country_levels <- c(
    country_levels,
    setdiff(
      all_countries,
      country_levels
    )
  )
}


# optional "Country (n = …)" axis labels
country_size <- cls_country_long %>%
  filter(Model == best_cls_model) %>%
  group_by(Country) %>%
  summarise(
    N_test = suppressWarnings(max(N_test, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(N_test = ifelse(is.finite(N_test), N_test, NA_real_))


# one label per country level (falls back to the plain country
# name when no sample-size column is available)
size_lookup <- setNames(
  country_size$N_test,
  country_size$Country
)


country_labels <- setNames(
  vapply(
    country_levels,
    
    function(cc) {
      
      nn <- if (cc %in% names(size_lookup)) {
        size_lookup[[cc]]
      } else {
        NA_real_
      }
      
      if (is.na(nn)) {
        cc
      } else {
        sprintf(
          "%s\n(n = %s)",
          cc,
          format(nn, big.mark = ",", trim = TRUE)
        )
      }
    },
    
    character(1)
  ),
  
  country_levels
)


n_countries <- length(country_levels)


apply_country_factor <- function(df) {
  
  df %>%
    mutate(
      Country = factor(
        Country,
        levels = country_levels
      )
    )
}


# ------------------------------------------------------------
# 6.3 Data for the Figure 5 panels
# ------------------------------------------------------------

best_cls <- cls_country_long %>%
  filter(Model == best_cls_model) %>%
  apply_country_factor() %>%
  mutate(
    Metric = factor(
      Metric,
      levels = cls_metrics
    )
  )


best_reg <- reg_country_long %>%
  filter(Model == best_reg_model) %>%
  apply_country_factor()


if (nrow(best_cls) == 0) {
  stop(
    "No country-level classification rows for model '",
    best_cls_model,
    "'. Check 'best_cls_model_manual'."
  )
}

if (nrow(best_reg) == 0) {
  stop(
    "No country-level regression rows for model '",
    best_reg_model,
    "'. Check 'best_reg_model_manual'."
  )
}


# mean across held-out countries (the value reported in the text)
mean_cls <- best_cls %>%
  group_by(Metric) %>%
  summarise(
    Overall = mean(Mean, na.rm = TRUE),
    .groups = "drop"
  )


mean_acc <- mean_cls$Overall[mean_cls$Metric == order_metric]

if (length(mean_acc) == 0 || is.na(mean_acc)) {
  mean_acc <- mean(
    best_cls$Mean[best_cls$Metric == "Accuracy"],
    na.rm = TRUE
  )
}


metric_colors <- c(
  "Accuracy"  = "#4C78A8",
  "Precision" = "#72B7B2",
  "Recall"    = "#E07B39",
  "F1"        = "#B07AA1"
)


# ============================================================
# 7. NEW FIGURE 5
#
# A : classification performance in each held-out country
# B : deviation from the mean (which countries stand out)
# C : correlation between predicted and observed fitness
# D : regression error (RMSE / MAE) in each held-out country
# ============================================================


# ------------------------------------------------------------
# 7A. Classification performance per country
# ------------------------------------------------------------

p_A <- ggplot(
  best_cls,
  aes(
    x = Country,
    y = Mean,
    fill = Metric
  )
) +
  
  geom_col(
    position = position_dodge(width = 0.82),
    width = 0.74
  ) +
  
  geom_errorbar(
    aes(
      ymin = CI_low,
      ymax = CI_high
    ),
    position = position_dodge(width = 0.82),
    width = 0.16,
    linewidth = 0.5,
    na.rm = TRUE
  ) +
  
  geom_hline(
    yintercept = mean_acc,
    linetype = "dashed",
    colour = "grey25",
    linewidth = 0.5
  ) +
  
  annotate(
    "text",
    x = n_countries + 0.45,
    y = mean_acc,
    label = sprintf(
      "mean %s = %.3f",
      order_metric,
      mean_acc
    ),
    hjust = 1,
    vjust = -0.6,
    size = 3,
    colour = "grey25"
  ) +
  
  scale_fill_manual(values = metric_colors) +
  
  scale_x_discrete(labels = country_labels) +
  
  coord_cartesian(ylim = c(0, 1.06)) +
  
  scale_y_continuous(
    breaks = seq(0, 1, by = 0.2)
  ) +
  
  labs(
    x = NULL,
    y = "Classification performance",
    fill = NULL,
    title = sprintf(
      "Classification, best model: %s",
      best_cls_model
    )
  ) +
  
  theme_classic(base_size = 12) +
  
  theme(
    axis.text.x = element_text(
      angle = 35,
      hjust = 1
    ),
    axis.title.y = element_text(face = "bold"),
    plot.title = element_text(size = 11, face = "bold"),
    legend.position = "top"
  )


if (show_chance_line) {
  
  p_A <- p_A +
    geom_hline(
      yintercept = 0.5,
      linetype = "dotted",
      colour = "grey55",
      linewidth = 0.45
    )
}


if (label_panel_values) {
  
  p_A <- p_A +
    geom_text(
      aes(label = sprintf("%.2f", Mean)),
      position = position_dodge(width = 0.82),
      vjust = -0.45,
      size = 2.1
    )
}


# ------------------------------------------------------------
# 7B. Deviation from the mean
#
# Directly answers "are some countries especially good or bad
# in terms of generalisability?"
# ------------------------------------------------------------

dev_df <- best_cls %>%
  filter(Metric == order_metric) %>%
  mutate(
    Delta = Mean - mean_acc
  )


dev_limit <- max(abs(dev_df$Delta), na.rm = TRUE) * 1.35

if (!is.finite(dev_limit) || dev_limit == 0) {
  dev_limit <- 0.1
}


p_B <- ggplot(
  dev_df,
  aes(
    x = Country,
    y = Delta,
    fill = Delta
  )
) +
  
  geom_col(width = 0.68) +
  
  geom_hline(
    yintercept = 0,
    colour = "grey25",
    linewidth = 0.5
  ) +
  
  geom_text(
    aes(
      label = sprintf("%+.3f", Delta),
      vjust = ifelse(Delta >= 0, -0.5, 1.3)
    ),
    size = 2.9
  ) +
  
  scale_fill_gradient2(
    low = "#B2182B",
    mid = "white",
    high = "#2166AC",
    midpoint = 0,
    guide = "none"
  ) +
  
  scale_x_discrete(labels = country_labels) +
  
  coord_cartesian(
    ylim = c(-dev_limit, dev_limit)
  ) +
  
  labs(
    x = NULL,
    y = sprintf(
      "%s \u2212 mean across countries",
      order_metric
    ),
    title = sprintf(
      "Per-country deviation (%s)",
      best_cls_model
    )
  ) +
  
  theme_classic(base_size = 12) +
  
  theme(
    axis.text.x = element_text(
      angle = 35,
      hjust = 1
    ),
    axis.title.y = element_text(face = "bold"),
    plot.title = element_text(size = 11, face = "bold")
  )


# ------------------------------------------------------------
# 7C. Correlation per country (Pearson r and R2)
# ------------------------------------------------------------

corr_df <- best_reg %>%
  filter(Metric %in% c("Pearson_r", "R2")) %>%
  mutate(
    Metric = factor(
      Metric,
      levels = c("Pearson_r", "R2"),
      labels = c("Pearson r", "R\u00B2")
    )
  )


corr_low <- min(c(0, corr_df$Mean, corr_df$CI_low), na.rm = TRUE)
corr_high <- max(c(1, corr_df$Mean, corr_df$CI_high), na.rm = TRUE)


p_C <- ggplot(
  corr_df,
  aes(
    x = Country,
    y = Mean,
    fill = Metric
  )
) +
  
  geom_col(
    position = position_dodge(width = 0.78),
    width = 0.70
  ) +
  
  geom_errorbar(
    aes(
      ymin = CI_low,
      ymax = CI_high
    ),
    position = position_dodge(width = 0.78),
    width = 0.16,
    linewidth = 0.5,
    na.rm = TRUE
  ) +
  
  geom_hline(
    yintercept = 0,
    colour = "grey25",
    linewidth = 0.45
  ) +
  
  scale_fill_manual(
    values = c(
      "Pearson r" = "#E07B39",
      "R\u00B2" = "#F2C185"
    )
  ) +
  
  scale_x_discrete(labels = country_labels) +
  
  coord_cartesian(
    ylim = c(
      min(corr_low * 1.15, 0),
      min(corr_high * 1.08, 1.08)
    )
  ) +
  
  labs(
    x = NULL,
    y = "Predicted vs observed fitness",
    fill = NULL,
    title = sprintf(
      "Regression, best model: %s",
      best_reg_model
    )
  ) +
  
  theme_classic(base_size = 12) +
  
  theme(
    axis.text.x = element_text(
      angle = 35,
      hjust = 1
    ),
    axis.title.y = element_text(face = "bold"),
    plot.title = element_text(size = 11, face = "bold"),
    legend.position = "top"
  )


# ------------------------------------------------------------
# 7D. Prediction error per country (RMSE and MAE)
# ------------------------------------------------------------

err_df <- best_reg %>%
  filter(Metric %in% c("RMSE", "MAE")) %>%
  mutate(
    Metric = factor(
      Metric,
      levels = c("RMSE", "MAE")
    )
  )


mean_rmse <- mean(
  err_df$Mean[err_df$Metric == "RMSE"],
  na.rm = TRUE
)


p_D <- ggplot(
  err_df,
  aes(
    x = Country,
    y = Mean,
    fill = Metric
  )
) +
  
  geom_col(
    position = position_dodge(width = 0.78),
    width = 0.70
  ) +
  
  geom_errorbar(
    aes(
      ymin = CI_low,
      ymax = CI_high
    ),
    position = position_dodge(width = 0.78),
    width = 0.16,
    linewidth = 0.5,
    na.rm = TRUE
  ) +
  
  scale_fill_manual(
    values = c(
      "RMSE" = "#2E8B57",
      "MAE" = "#9BCFB0"
    )
  ) +
  
  scale_x_discrete(labels = country_labels) +
  
  labs(
    x = NULL,
    y = "Prediction error (lower is better)",
    fill = NULL,
    title = sprintf(
      "Prediction error, %s",
      best_reg_model
    )
  ) +
  
  theme_classic(base_size = 12) +
  
  theme(
    axis.text.x = element_text(
      angle = 35,
      hjust = 1
    ),
    axis.title.y = element_text(face = "bold"),
    plot.title = element_text(size = 11, face = "bold"),
    legend.position = "top"
  )


# mean RMSE reference line (only if RMSE is available)
if (is.finite(mean_rmse)) {
  
  p_D <- p_D +
    
    geom_hline(
      yintercept = mean_rmse,
      linetype = "dashed",
      colour = "grey25",
      linewidth = 0.5
    ) +
    
    annotate(
      "text",
      x = n_countries + 0.45,
      y = mean_rmse,
      label = sprintf("mean RMSE = %.2f", mean_rmse),
      hjust = 1,
      vjust = -0.6,
      size = 3,
      colour = "grey25"
    )
}


# ------------------------------------------------------------
# Assemble Figure 5
# ------------------------------------------------------------

figure5 <-
  (p_A | p_B) /
  (p_C | p_D) +
  plot_annotation(
    tag_levels = "A"
  ) &
  theme(
    plot.tag = element_text(
      size = 16,
      face = "bold"
    )
  )


print(figure5)


ggsave(
  file.path(
    output_dir,
    "Figure5_country_level_best_model.png"
  ),
  figure5,
  width = figure5_width,
  height = figure5_height,
  dpi = 600
)


ggsave(
  file.path(
    output_dir,
    "Figure5_country_level_best_model.pdf"
  ),
  figure5,
  width = figure5_width,
  height = figure5_height
)


# ------------------------------------------------------------
# Classification-only version of Figure 5
#
# Use this if the journal prefers a single-panel figure that
# matches the reviewer request literally
# ("a bar chart for each country ... for the best model").
# ------------------------------------------------------------

figure5_simple <- p_A +
  labs(
    title = sprintf(
      "Leave-one-country-out performance of the best model (%s)",
      best_cls_model
    )
  )


ggsave(
  file.path(
    output_dir,
    "Figure5_country_level_classification_only.png"
  ),
  figure5_simple,
  width = 10,
  height = 5.5,
  dpi = 600
)


ggsave(
  file.path(
    output_dir,
    "Figure5_country_level_classification_only.pdf"
  ),
  figure5_simple,
  width = 10,
  height = 5.5
)


# ============================================================
# 8. Per-country robustness across sensitivity analyses
#
# Supplementary: is a "bad" country bad in every country set,
# or only in one particular analysis?
# ============================================================

sens_country <- cls_country_all %>%
  filter(Model == best_cls_model) %>%
  country_to_long(cls_metrics) %>%
  filter(Metric == order_metric) %>%
  apply_country_factor()


p_country_scenario <- ggplot(
  sens_country,
  aes(
    x = Country,
    y = Mean,
    fill = Scenario
  )
) +
  
  geom_col(
    position = position_dodge(width = 0.82),
    width = 0.74
  ) +
  
  scale_x_discrete(labels = country_labels) +
  
  coord_cartesian(ylim = c(0, 1.05)) +
  
  scale_fill_brewer(palette = "Blues") +
  
  labs(
    x = NULL,
    y = order_metric,
    fill = NULL,
    title = sprintf(
      "Per-country %s of %s across country sets",
      order_metric,
      best_cls_model
    )
  ) +
  
  theme_classic(base_size = 12) +
  
  theme(
    axis.text.x = element_text(
      angle = 35,
      hjust = 1
    ),
    axis.title.y = element_text(face = "bold"),
    plot.title = element_text(size = 11, face = "bold"),
    legend.position = "top"
  )


print(p_country_scenario)


ggsave(
  file.path(
    output_dir,
    "Supplementary_country_level_across_scenarios.png"
  ),
  p_country_scenario,
  width = 11,
  height = 5.5,
  dpi = 600
)


ggsave(
  file.path(
    output_dir,
    "Supplementary_country_level_across_scenarios.pdf"
  ),
  p_country_scenario,
  width = 11,
  height = 5.5
)


# ============================================================
# 9. SUPPLEMENTARY FIGURE
#    (this was the previous Figure 5)
#
# Panel A : model-level classification means with 95% CI
# Panels B-E : sensitivity-analysis heatmaps
# ============================================================

main_cls <- cls_summary_all %>%
  filter(Scenario == "Main analysis")


main_cls_long <- bind_rows(
  
  main_cls %>%
    transmute(
      Model,
      Metric = "Accuracy",
      Mean = Accuracy_mean,
      CI_low = Accuracy_ci95_low,
      CI_high = Accuracy_ci95_high
    ),
  
  main_cls %>%
    transmute(
      Model,
      Metric = "Precision",
      Mean = Precision_mean,
      CI_low = Precision_ci95_low,
      CI_high = Precision_ci95_high
    ),
  
  main_cls %>%
    transmute(
      Model,
      Metric = "Recall",
      Mean = Recall_mean,
      CI_low = Recall_ci95_low,
      CI_high = Recall_ci95_high
    ),
  
  main_cls %>%
    transmute(
      Model,
      Metric = "F1",
      Mean = F1_mean,
      CI_low = F1_ci95_low,
      CI_high = F1_ci95_high
    )
)


main_cls_long <- main_cls_long %>%
  mutate(
    Model = factor(Model, levels = model_order),
    Metric = factor(
      Metric,
      levels = c("Accuracy", "Precision", "Recall", "F1")
    )
  )


p_main_cls <- ggplot(
  main_cls_long,
  aes(
    x = Model,
    y = Mean,
    fill = Metric
  )
) +
  
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.70
  ) +
  
  geom_errorbar(
    aes(
      ymin = CI_low,
      ymax = CI_high
    ),
    position = position_dodge(width = 0.8),
    width = 0.18,
    linewidth = 0.65
  ) +
  
  geom_text(
    aes(
      label = sprintf("%.3f", Mean)
    ),
    position = position_dodge(width = 0.8),
    vjust = -0.55,
    size = 3.2
  ) +
  
  scale_fill_manual(values = metric_colors) +
  
  coord_cartesian(
    ylim = c(0, 1.03)
  ) +
  
  scale_y_continuous(
    breaks = seq(0, 1, by = 0.2)
  ) +
  
  labs(
    x = NULL,
    y = "Classification performance",
    fill = NULL
  ) +
  
  theme_classic(base_size = 13) +
  
  theme(
    axis.text.x = element_text(
      angle = 35,
      hjust = 1
    ),
    
    axis.title.y = element_text(
      face = "bold"
    ),
    
    legend.position = "top"
  )


# ------------------------------------------------------------
# 9A. Accuracy heatmap
# ------------------------------------------------------------

accuracy_heat <- cls_summary_all %>%
  select(
    Model,
    Scenario,
    Accuracy_mean
  )


p_heat_accuracy <- ggplot(
  accuracy_heat,
  aes(
    x = Scenario,
    y = Model,
    fill = Accuracy_mean
  )
) +
  geom_tile(
    linewidth = 0.7,
    color = "white"
  ) +
  geom_text(
    aes(
      label = sprintf("%.3f", Accuracy_mean)
    ),
    size = 3.8
  ) +
  scale_fill_gradient(
    low = "#F3F6FA",
    high = "#4C78A8"
  ) +
  labs(
    x = NULL,
    y = NULL,
    fill = "Accuracy"
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(
      angle = 30,
      hjust = 1
    ),
    axis.line = element_blank(),
    axis.ticks = element_blank()
  )


# ------------------------------------------------------------
# 9B. RMSE heatmap (lower = darker = better)
# ------------------------------------------------------------

rmse_heat <- reg_summary_all %>%
  select(
    Model,
    Scenario,
    RMSE_mean
  )


p_heat_rmse <- ggplot(
  rmse_heat,
  aes(
    x = Scenario,
    y = Model,
    fill = RMSE_mean
  )
) +
  geom_tile(
    linewidth = 0.7,
    color = "white"
  ) +
  geom_text(
    aes(
      label = sprintf("%.2f", RMSE_mean)
    ),
    size = 3.8
  ) +
  scale_fill_gradient(
    low = "#2E8B57",
    high = "#F2F7F2"
  ) +
  labs(
    x = NULL,
    y = NULL,
    fill = "RMSE"
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(
      angle = 30,
      hjust = 1
    ),
    axis.line = element_blank(),
    axis.ticks = element_blank()
  )


# ------------------------------------------------------------
# 9C. MAE heatmap
# ------------------------------------------------------------

mae_heat <- reg_summary_all %>%
  select(
    Model,
    Scenario,
    MAE_mean
  )


p_heat_mae <- ggplot(
  mae_heat,
  aes(
    x = Scenario,
    y = Model,
    fill = MAE_mean
  )
) +
  geom_tile(
    linewidth = 0.7,
    color = "white"
  ) +
  geom_text(
    aes(
      label = sprintf("%.2f", MAE_mean)
    ),
    size = 3.8
  ) +
  scale_fill_gradient(
    low = "#B07AA1",
    high = "#F7F2F6"
  ) +
  labs(
    x = NULL,
    y = NULL,
    fill = "MAE"
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(
      angle = 30,
      hjust = 1
    ),
    axis.line = element_blank(),
    axis.ticks = element_blank()
  )


# ------------------------------------------------------------
# 9D. Pearson r heatmap
# ------------------------------------------------------------

pearson_heat <- reg_summary_all %>%
  select(
    Model,
    Scenario,
    Pearson_r_mean
  )


p_heat_pearson <- ggplot(
  pearson_heat,
  aes(
    x = Scenario,
    y = Model,
    fill = Pearson_r_mean
  )
) +
  geom_tile(
    linewidth = 0.7,
    color = "white"
  ) +
  geom_text(
    aes(
      label = sprintf("%.3f", Pearson_r_mean)
    ),
    size = 3.8
  ) +
  scale_fill_gradient(
    low = "#FFF4E6",
    high = "#E07B39"
  ) +
  labs(
    x = NULL,
    y = NULL,
    fill = "Pearson r"
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(
      angle = 30,
      hjust = 1
    ),
    axis.line = element_blank(),
    axis.ticks = element_blank()
  )


sensitivity_figure <-
  (p_heat_accuracy | p_heat_rmse) /
  (p_heat_mae | p_heat_pearson) +
  plot_annotation(
    tag_levels = "A"
  )


supplementary_model_level <- p_main_cls / sensitivity_figure +
  plot_layout(heights = c(1, 1.4)) +
  plot_annotation(
    tag_levels = "A"
  ) &
  theme(
    plot.tag = element_text(
      size = 18,
      face = "bold"
    )
  )


print(supplementary_model_level)


ggsave(
  file.path(
    output_dir,
    "Supplementary_model_level_main_and_sensitivity.png"
  ),
  supplementary_model_level,
  width = 14,
  height = 12,
  dpi = 600
)


ggsave(
  file.path(
    output_dir,
    "Supplementary_model_level_main_and_sensitivity.pdf"
  ),
  supplementary_model_level,
  width = 14,
  height = 12
)


# ============================================================
# 10. SUPPLEMENTARY: all models x all countries
# ============================================================

cls_country_main <- cls_country_all %>%
  filter(Scenario == "Main analysis") %>%
  apply_country_factor()


reg_country_main <- reg_country_all %>%
  filter(Scenario == "Main analysis") %>%
  apply_country_factor()


p_country_accuracy <- ggplot(
  cls_country_main,
  aes(
    x = Country,
    y = Model,
    fill = Accuracy
  )
) +
  
  geom_tile(
    color = "white",
    linewidth = 0.4
  ) +
  
  geom_text(
    aes(
      label = sprintf(
        "%.2f",
        Accuracy
      )
    ),
    size = 2.6
  ) +
  
  scale_fill_gradient(
    low = "white",
    high = "#4C78A8"
  ) +
  
  labs(
    x = "Held-out country",
    y = NULL,
    fill = "Accuracy"
  ) +
  
  theme_classic(
    base_size = 11
  ) +
  
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    ),
    
    axis.line = element_blank(),
    axis.ticks = element_blank()
  )


p_country_r2 <- ggplot(
  reg_country_main,
  aes(
    x = Country,
    y = Model,
    fill = R2
  )
) +
  
  geom_tile(
    color = "white",
    linewidth = 0.4
  ) +
  
  geom_text(
    aes(
      label = sprintf(
        "%.2f",
        R2
      )
    ),
    size = 2.6
  ) +
  
  scale_fill_gradient2(
    low = "#B2182B",
    mid = "white",
    high = "#2166AC",
    midpoint = 0
  ) +
  
  labs(
    x = "Held-out country",
    y = NULL,
    fill = expression(R^2)
  ) +
  
  theme_classic(
    base_size = 11
  ) +
  
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    ),
    
    axis.line = element_blank(),
    axis.ticks = element_blank()
  )


country_figure <- p_country_accuracy +
  p_country_r2 +
  plot_annotation(
    tag_levels = "A"
  )


print(country_figure)


ggsave(
  file.path(
    output_dir,
    "Supplementary_country_level_LOCO_all_models.png"
  ),
  country_figure,
  width = 16,
  height = 7,
  dpi = 600
)


ggsave(
  file.path(
    output_dir,
    "Supplementary_country_level_LOCO_all_models.pdf"
  ),
  country_figure,
  width = 16,
  height = 7
)


# ============================================================
# 11. SUPPLEMENTARY TABLES
# ============================================================

# ------------------------------------------------------------
# 11.1 Model-level mean (95% CI) across scenarios
# ------------------------------------------------------------

cls_long <- bind_rows(
  
  cls_summary_all %>%
    transmute(
      Model,
      Scenario,
      Task = "Classification",
      Metric = "Accuracy",
      Mean = Accuracy_mean,
      CI_low = Accuracy_ci95_low,
      CI_high = Accuracy_ci95_high
    ),
  
  cls_summary_all %>%
    transmute(
      Model,
      Scenario,
      Task = "Classification",
      Metric = "Precision",
      Mean = Precision_mean,
      CI_low = Precision_ci95_low,
      CI_high = Precision_ci95_high
    ),
  
  cls_summary_all %>%
    transmute(
      Model,
      Scenario,
      Task = "Classification",
      Metric = "Recall",
      Mean = Recall_mean,
      CI_low = Recall_ci95_low,
      CI_high = Recall_ci95_high
    ),
  
  cls_summary_all %>%
    transmute(
      Model,
      Scenario,
      Task = "Classification",
      Metric = "F1",
      Mean = F1_mean,
      CI_low = F1_ci95_low,
      CI_high = F1_ci95_high
    )
)


reg_long <- bind_rows(
  
  reg_summary_all %>%
    transmute(
      Model,
      Scenario,
      Task = "Regression",
      Metric = "RMSE",
      Mean = RMSE_mean,
      CI_low = RMSE_ci95_low,
      CI_high = RMSE_ci95_high
    ),
  
  reg_summary_all %>%
    transmute(
      Model,
      Scenario,
      Task = "Regression",
      Metric = "MAE",
      Mean = MAE_mean,
      CI_low = MAE_ci95_low,
      CI_high = MAE_ci95_high
    ),
  
  reg_summary_all %>%
    transmute(
      Model,
      Scenario,
      Task = "Regression",
      Metric = "R2",
      Mean = R2_mean,
      CI_low = R2_ci95_low,
      CI_high = R2_ci95_high
    ),
  
  reg_summary_all %>%
    transmute(
      Model,
      Scenario,
      Task = "Regression",
      Metric = "Pearson r",
      Mean = Pearson_r_mean,
      CI_low = Pearson_r_ci95_low,
      CI_high = Pearson_r_ci95_high
    )
)


supp_long <- bind_rows(
  cls_long,
  reg_long
) %>%
  
  mutate(
    
    Result = sprintf(
      "%.3f (%.3f, %.3f)",
      Mean,
      CI_low,
      CI_high
    ),
    
    Model = factor(
      Model,
      levels = model_order
    ),
    
    Scenario = factor(
      Scenario,
      levels = scenario_order
    )
  )


supp_table <- supp_long %>%
  
  select(
    Model,
    Task,
    Metric,
    Scenario,
    Result
  ) %>%
  
  pivot_wider(
    names_from = Scenario,
    values_from = Result
  ) %>%
  
  arrange(
    Model,
    Task,
    Metric
  )


# ------------------------------------------------------------
# 11.2 NEW: country-level table for the best model
# ------------------------------------------------------------

country_best_table <- bind_rows(
  best_cls %>%
    mutate(
      Task = "Classification",
      Model = best_cls_model
    ),
  best_reg %>%
    mutate(
      Task = "Regression",
      Model = best_reg_model
    )
) %>%
  
  mutate(
    Value = round(Mean, 3)
  ) %>%
  
  select(
    Country,
    Task,
    Model,
    Metric,
    Value,
    N_test
  ) %>%
  
  pivot_wider(
    names_from = Metric,
    values_from = Value
  ) %>%
  
  arrange(Task, Country)


# ------------------------------------------------------------
# 11.3 NEW: country-level table, all models
# ------------------------------------------------------------

country_all_table <- bind_rows(
  cls_country_long %>% mutate(Task = "Classification"),
  reg_country_long %>% mutate(Task = "Regression")
) %>%
  
  mutate(Value = round(Mean, 3)) %>%
  
  select(
    Country,
    Model,
    Task,
    Metric,
    Value
  ) %>%
  
  pivot_wider(
    names_from = Metric,
    values_from = Value
  ) %>%
  
  arrange(Task, Model, Country)


# ------------------------------------------------------------
# 11.4 Pooled overall performance
# ------------------------------------------------------------

pooled_cls <- cls_overall_all %>%
  
  transmute(
    Model,
    Scenario,
    Accuracy,
    Precision,
    Recall,
    F1
  )


pooled_reg <- reg_overall_all %>%
  
  transmute(
    Model,
    Scenario,
    RMSE,
    MAE,
    R2,
    Pearson_r
  )


pooled_all <- full_join(
  pooled_cls,
  pooled_reg,
  by = c(
    "Model",
    "Scenario"
  )
) %>%
  
  mutate(
    across(
      c(
        Accuracy,
        Precision,
        Recall,
        F1,
        RMSE,
        MAE,
        R2,
        Pearson_r
      ),
      ~ round(.x, 3)
    )
  ) %>%
  
  arrange(
    Model,
    Scenario
  )


write_xlsx(
  
  list(
    
    "Country_best_model" = country_best_table,
    
    "Country_all_models" = country_all_table,
    
    "Model_mean_95CI" = supp_table,
    
    "Pooled_overall" = pooled_all,
    
    "Classification_summary_raw" =
      cls_summary_all,
    
    "Regression_summary_raw" =
      reg_summary_all
    
  ),
  
  file.path(
    output_dir,
    "Supplementary_ML_performance_tables.xlsx"
  )
)


write_csv(
  country_best_table,
  file.path(
    output_dir,
    "Figure5_country_level_best_model_values.csv"
  )
)


write_csv(
  supp_table,
  file.path(
    output_dir,
    "Supplementary_ML_performance_mean_95CI.csv"
  )
)


# ============================================================
# 12. TEXT SUMMARY
#
# Numbers to paste into the Results text, the figure legend
# and the response to the reviewer.
# ============================================================

summarise_spread <- function(df, metric, label) {
  
  d <- df %>%
    filter(Metric == metric) %>%
    filter(!is.na(Mean))
  
  if (nrow(d) == 0) {
    return(invisible(NULL))
  }
  
  best_row <- d[which.max(d$Mean), ]
  worst_row <- d[which.min(d$Mean), ]
  
  cat(
    sprintf(
      "%-12s mean = %.3f | range %.3f-%.3f | SD = %.3f | CV = %.1f%%\n",
      label,
      mean(d$Mean),
      min(d$Mean),
      max(d$Mean),
      sd(d$Mean),
      100 * sd(d$Mean) / mean(d$Mean)
    )
  )
  
  cat(
    sprintf(
      "             highest: %s (%.3f) | lowest: %s (%.3f)\n",
      as.character(best_row$Country),
      best_row$Mean,
      as.character(worst_row$Country),
      worst_row$Mean
    )
  )
}


cat("\n============================================================\n")
cat("COUNTRY-LEVEL GENERALISABILITY SUMMARY (main analysis)\n")
cat("============================================================\n")

cat(
  "\nClassification model: ", best_cls_model,
  "\nRegression model:     ", best_reg_model,
  "\nHeld-out countries:   ", length(country_levels),
  " (", paste(country_levels, collapse = ", "), ")\n\n",
  sep = ""
)

for (m in cls_metrics) {
  summarise_spread(best_cls, m, m)
}

cat("\n")

for (m in c("Pearson_r", "R2", "RMSE", "MAE")) {
  summarise_spread(best_reg, m, m)
}

cat("\nCountry order used in Figure 5 (best -> worst by ")
cat(order_metric, "):\n", sep = "")
print(country_levels)

cat("\nPer-country values of the best model:\n")
print(as.data.frame(country_best_table), row.names = FALSE)


# ============================================================
# 13. Finished
# ============================================================

cat(
  "\nAll figures and tables saved to:\n",
  output_dir,
  "\n"
)


# ============================================================
# Suggested new Figure 5 legend
# ------------------------------------------------------------
# Figure 5. Country-level cross-country generalizability of the
# best-performing machine-learning model in leave-one-country-out
# validation. (A) Classification performance (accuracy,
# precision, recall and F1 score) of the best classification
# model in each held-out country. The dashed line indicates the
# mean accuracy across held-out countries and the dotted line
# indicates the 0.5 chance level. (B) Difference between the
# accuracy in each held-out country and the mean accuracy across
# countries; positive values (blue) indicate countries predicted
# better than average and negative values (red) indicate
# countries predicted worse than average. (C) Pearson
# correlation coefficient and R2 between predicted and observed
# fitness in each held-out country for the best regression
# model. (D) Root mean squared error and mean absolute error in
# each held-out country. In all panels countries are ordered
# from the highest to the lowest classification accuracy, and
# each bar represents the single held-out fold for that country.
# ============================================================