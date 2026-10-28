# ============================================================
# SARS-CoV-2 fitness prediction
# LOCO model performance visualization
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
    "SVR" = "SVM"
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


# ============================================================
# 6. MAIN FIGURE
#
# Single Panel:
# Accuracy + Precision + Recall + F1
# (mean across held-out countries with 95% CI)
# ============================================================

main_cls <- cls_summary_all %>%
  filter(Scenario == "Main analysis")


# ------------------------------------------------------------
# Convert to long format for 4 classification metrics
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Main plot
# ------------------------------------------------------------

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


print(p_main_cls)


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

ggsave(
  file.path(
    output_dir,
    "Figure_Main_classification_performance_4metrics.png"
  ),
  p_main_cls,
  width = 11,
  height = 6,
  dpi = 600
)

ggsave(
  file.path(
    output_dir,
    "Figure_Main_classification_performance_4metrics.pdf"
  ),
  p_main_cls,
  width = 11,
  height = 6
)


# ============================================================
# 7. SENSITIVITY ANALYSIS HEATMAP
#
# A: Accuracy
# B: RMSE
# C: MAE
# D: Pearson r
#
# Accuracy / Pearson r: higher = better
# RMSE / MAE: lower = better
# ============================================================


# ------------------------------------------------------------
# 7A. Accuracy
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
# 7B. RMSE
#
# Lower RMSE = better
# Therefore lower values are shown with darker colors
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
# 7C. MAE
#
# Lower MAE = better
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
# 7D. Pearson r
#
# Higher Pearson r = better
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


# ============================================================
# Combine four panels
# ============================================================

sensitivity_figure <-
  (p_heat_accuracy | p_heat_rmse) /
  (p_heat_mae | p_heat_pearson) +
  plot_annotation(
    tag_levels = "A"
  )


print(sensitivity_figure)




# ============================================================
# Save
# ============================================================

ggsave(
  file.path(
    output_dir,
    "Figure_Sensitivity_analysis_heatmap_4metrics.png"
  ),
  sensitivity_figure,
  width = 9,
  height = 6,
  dpi = 600
)


ggsave(
  file.path(
    output_dir,
    "Figure_Sensitivity_analysis_heatmap_4metrics.pdf"
  ),
  sensitivity_figure,
  width = 9,
  height = 6
)


combined_figure <- p_main_cls / sensitivity_figure +
  plot_layout(heights = c(1, 1.4)) +  plot_annotation(
    tag_levels = "A"
  ) &
  theme(
    plot.tag = element_text(
      size = 18,
      face = "bold"
    )
  )

print(combined_figure)

ggsave(
  file.path(output_dir, "Figure_combined_main_and_sensitivity.png"),
  combined_figure,
  width = 14,
  height = 12,
  dpi = 600
)

ggsave(
  file.path(output_dir, "Figure_combined_main_and_sensitivity.pdf"),
  combined_figure,
  width = 14,
  height = 12
)

# ============================================================
# 8. COMPLETE SUPPLEMENTARY TABLE
#
# Format:
#
# Model
# Metric
# Main analysis
# Sensitivity 1
# Sensitivity 2
# Sensitivity 3
#
# Values = mean (95% CI)
# ============================================================


# ------------------------------------------------------------
# Classification table
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


# ------------------------------------------------------------
# Regression table
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

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


print(
  supp_table,
  n = Inf
)


# ============================================================
# 9. POOLED OVERALL PERFORMANCE TABLE
#
# These are the pooled sample-level estimates
# from *_metrics_overall_loco.csv
# ============================================================


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


# ============================================================
# 10. Save supplementary Excel workbook
# ============================================================

write_xlsx(
  
  list(
    
    "Mean_95CI" = supp_table,
    
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


# Also save CSV
write_csv(
  supp_table,
  file.path(
    output_dir,
    "Supplementary_ML_performance_mean_95CI.csv"
  )
)


# ============================================================
# 11. COUNTRY-LEVEL PERFORMANCE
#
# Main analysis only
# Supplementary Figure
# ============================================================


main_folder <- file.path(
  base_dir,
  "loco_results_sens1"
)


cls_country <- read_csv(
  
  file.path(
    main_folder,
    "classification_metrics_by_country_loco.csv"
  ),
  
  show_col_types = FALSE
  
) %>%
  
  mutate(
    Model = standardize_model(Model),
    Model = factor(
      Model,
      levels = model_order
    )
  )


reg_country <- read_csv(
  
  file.path(
    main_folder,
    "regression_metrics_by_country_loco.csv"
  ),
  
  show_col_types = FALSE
  
) %>%
  
  mutate(
    Model = standardize_model(Model),
    Model = factor(
      Model,
      levels = model_order
    )
  )


# ------------------------------------------------------------
# Country-level Accuracy
# ------------------------------------------------------------

p_country_accuracy <- ggplot(
  cls_country,
  aes(
    x = HeldOutCountry,
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


# ------------------------------------------------------------
# Country-level R2
# ------------------------------------------------------------

p_country_r2 <- ggplot(
  reg_country,
  aes(
    x = HeldOutCountry,
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
    "Supplementary_country_level_LOCO_performance.png"
  ),
  country_figure,
  width = 16,
  height = 7,
  dpi = 600
)


ggsave(
  file.path(
    output_dir,
    "Supplementary_country_level_LOCO_performance.pdf"
  ),
  country_figure,
  width = 16,
  height = 7
)


# ============================================================
# Finished
# ============================================================

cat(
  "\nAll figures and tables saved to:\n",
  output_dir,
  "\n"
)

