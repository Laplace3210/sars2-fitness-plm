

library(ggplot2)
library(dplyr)
library(readr)

# --- Paths -------------------------------------------------------------------
# Run this script from its own directory (figures/). The per-model
# "<model>_obs_vs_pred.csv" tables are exported from the leave-one-country-out
# predictions written by ../code/04_fitness_models/; point RESULTS_DIR at
# whichever analysis folder holds them.
base_dir <- Sys.getenv("RESULTS_DIR", "../results/main_analysis/")
save_dir <- Sys.getenv("FIGURE_DIR", "../results/figures/")
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
# -----------------------------------------------------------------------------


plot_facet_scatter <- function(csv_path, model_name = "Model", save_path = NULL) {
  dat <- read_csv(csv_path)
  dat$Country <- factor(dat$Country, levels = unique(dat$Country))
  cor_df <- dat %>%
    group_by(Country) %>%
    summarise(r = cor(Predicted, Observed, method = "pearson", use = "complete.obs"))
  dat <- left_join(dat, cor_df, by = "Country")
  p <- ggplot(dat, aes(x = Predicted, y = Observed)) +
    geom_point(alpha = 0.7, color = "palevioletred2", size = 2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 1) +
    facet_wrap(~ Country, scales = "free") +
    geom_text(
      data = cor_df, 
      aes(x = -Inf, y = Inf, label = sprintf("r = %.3f", r)), 
      hjust = -0.1, vjust = 1.2, size = 4.5, fontface = "bold", color = "#1b7837",
      inherit.aes = FALSE
    ) +
    theme_bw(base_size = 14) +
    labs(
      x = paste(model_name, "predicted fitness"), 
      y = "Observed fitness",
      title = paste("Predict vs Observed Fitness by Country -", model_name)
    ) +
    theme(
      axis.text = element_text(size = 12),
      strip.text = element_text(size = 13, face = "bold"),
      panel.grid.major = element_line(color = "gray80", linetype = "dashed"),
      panel.grid.minor = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 17, face = "bold")
    )
  if(!is.null(save_path)) {
    ggsave(save_path, p, width = 10, height = 7, dpi = 300)
  }
  return(p)
}

# 示例调用
# p_xgb <- plot_facet_scatter(
#   file.path(base_dir, "XGBoost_obs_vs_pred.csv"), 
#   model_name = "XGBoost", 
#   save_path = file.path(save_dir, "XGBoost_scatter_facet.png")
# )
# print(p_xgb)

models <- c("XGBoost", "Stepwise GAM", "Random Forest", "Ridge"
            #, "SVR"
)

for(model in models){
  csv_path <- paste0(base_dir, model, "_obs_vs_pred.csv")
  save_path <- paste0(save_dir, model, "_scatter_facet.png")
  if(file.exists(csv_path)){
    plot_facet_scatter(csv_path, model, save_path)
  }
}


library(ggplot2)
library(readr)

plot_scatter_overall <- function(csv_path, model_name = "Model", save_path = NULL) {
  dat <- read_csv(csv_path)
  # 计算全体pearson相关系数
  r <- cor(dat$Predicted, dat$Observed, method = "pearson", use = "complete.obs")
  p <- ggplot(dat, aes(x = Predicted, y = Observed)) +
    geom_point(alpha = 0.75, color = "palevioletred2", size = 2.5) +
    scale_x_continuous(breaks = seq(0, 150, by = 50), limits = c(0, 150)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 1) +
    # annotate("text", x = Inf, y = -Inf, 
    #          label = sprintf("r = %.3f", r),
    #          hjust = 1.1, vjust = -1.1, size = 6, fontface = "bold", color = "#1b7837") +
    theme_bw(base_size = 15) +
    labs(
      x = paste(model_name, "predicted fitness"), 
      y = "Observed fitness",
      title =  model_name
    ) +
    theme(
      axis.text = element_text(size = 13),
      panel.grid.major = element_line(color = "gray80", linetype = "dashed"),
      panel.grid.minor = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 15, face = "bold")
    )
  if(!is.null(save_path)) {
    ggsave(save_path, p, width = 7, height = 6, dpi = 450)
  }
  return(p)
}

# 示例调用
# p_xgb_all <- plot_scatter_overall(
#   file.path(base_dir, "XGBoost_obs_vs_pred.csv"), 
#   model_name = "XGBoost", 
#   save_path = file.path(save_dir, "XGBoost_scatter_overall.png")
# )
# print(p_xgb_all)


for(model in models){
  csv_path <- paste0(base_dir, model, "_obs_vs_pred.csv")
  save_path <- paste0(save_dir, model, "_scatter_overall.png")
  if(file.exists(csv_path)){
    plot_scatter_overall(csv_path, model, save_path)
  }
}

library(ggplot2)
library(patchwork)

# 先把每个模型的ggplot对象都存下来
plots <- lapply(models, function(model){
  csv_path <- paste0(base_dir, model, "_obs_vs_pred.csv")
  if(file.exists(csv_path)){
    plot_scatter_overall(csv_path, model)
  } else {
    NULL
  }
})

library(cowplot)
plots[[1]] <- plots[[1]] + annotate("text", x = Inf, y = -Inf, 
                                   label = sprintf("r = %.3f", 0.895),
                                   hjust = 1.1, vjust = -1.1, size = 6, fontface = "bold", color = "#1b7837")
plots[[2]] <- plots[[2]] + annotate("text", x = Inf, y = -Inf, 
                                    label = sprintf("r = %.3f", 0.683),
                                    hjust = 1.1, vjust = -1.1, size = 6, fontface = "bold", color = "#1b7837")
plots[[3]] <- plots[[3]] + annotate("text", x = Inf, y = -Inf, 
                                    label = sprintf("r = %.3f", 0.878),
                                    hjust = 1.1, vjust = -1.1, size = 6, fontface = "bold", color = "#1b7837")
plots[[4]] <- plots[[4]] + annotate("text", x = Inf, y = -Inf, 
                                    label = sprintf("r = %.3f", 0.535),
                                    hjust = 1.1, vjust = -1.1, size = 6, fontface = "bold", color = "#1b7837")
panel <- plot_grid(
  plots[[1]], plots[[2]], plots[[3]],
  plots[[4]], labels="AUTO",
  #plots[[5]], NULL,
  nrow = 2, ncol = 2,label_size = 16
)
print(panel)
ggsave(paste0(save_dir, "all_models_scatter_overall_panel.png"),
       panel, width = 9, height = 7, dpi = 450)


###############计算总体与分国家四项指标(RMSE、MAE、R²、Pearson r)###########

library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(Metrics) # 用于rmse和mae

# 计算四项指标
get_metrics <- function(df) {
  rmse_val <- rmse(df$Observed, df$Predicted)
  mae_val <- mae(df$Observed, df$Predicted)
  r2_val <- 1 - sum((df$Observed - df$Predicted)^2)/sum((df$Observed - mean(df$Observed))^2)
  r_val <- cor(df$Observed, df$Predicted, use = "complete.obs")
  tibble(RMSE = rmse_val, MAE = mae_val, R2 = r2_val, Pearson_r = r_val)
}


library(purrr)

models <- c("XGBoost", "Stepwise_GAM", "Random_Forest", "Ridge", "SVR")


# 总体指标
overall_metrics <- map_dfr(models, function(model){
  csv_path <- paste0(base_dir, model, "_obs_vs_pred.csv")
  if(file.exists(csv_path)){
    df <- read_csv(csv_path)
    m <- get_metrics(df)
    m$Model <- model
    m
  }
}, .id = "ID") %>% select(Model, everything(), -ID)

write_csv(overall_metrics, paste0(base_dir, "all_models_country_metrics.csv"))




# 分国家指标
country_metrics <- map_dfr(models, function(model){
  csv_path <- paste0(base_dir, model, "_obs_vs_pred.csv")
  if(file.exists(csv_path)){
    df <- read_csv(csv_path)
    df %>%
      group_by(Country) %>%
      do(get_metrics(.)) %>%
      mutate(Model = model)
  }
}) %>% select(Model, Country, everything())


# library(reshape2)
# 
# # 把数据转成长格式便于绘图
# overall_long <- overall_metrics %>%
#   pivot_longer(cols = c("RMSE", "MAE", "R2", "Pearson_r"),
#                names_to = "Metric", values_to = "Value")
# 
# ggplot(overall_long, aes(x = Model, y = Value, fill = Metric)) +
#   geom_bar(stat = "identity", position = "dodge") +
#   scale_fill_brewer(palette = "Paired") +
#   theme_bw(base_size = 15) +
#   labs(title = "Overall Model Performance", x = NULL, y = NULL) +
#   facet_wrap(~Metric, scales = "free_y", ncol = 2)

library(ggplot2)
library(tidyr)
library(forcats)

# 长格式
overall_long <- overall_metrics %>%
  pivot_longer(cols = c("RMSE", "MAE", "R2", "Pearson_r"),
               names_to = "Metric", values_to = "Value")

max_rmse <- max(overall_metrics$RMSE)

ggplot() +
  # 柱状：RMSE/MAE
  geom_bar(data = overall_long %>% filter(Metric %in% c("RMSE", "MAE")),
           aes(x = fct_inorder(Model), y = Value, fill = Metric),
           stat = "identity", position = "dodge", width = 0.6) +
  # 线图：R2/Pearson r
  geom_line(data = overall_long %>% filter(Metric %in% c("R2", "Pearson_r")),
            aes(x = fct_inorder(Model), y = Value * max_rmse, color = Metric, group = Metric),
            size = 1.2) +
  scale_y_continuous(
    name = "RMSE / MAE",
    sec.axis = sec_axis(~./max_rmse, name = "R2 / Pearson r")
  ) +
  scale_color_manual(
    name = "",    # 新legend标题
    values = c("Pearson_r" = "red", "R2" = "peachpuff"),
    labels = c("Pearson_r" = "Pearson r", "R2" = "R2")
  )+
  theme_bw(base_size = 13) +
  labs(#title = model,
    x = "Country", y = NULL, fill = "Bar", color = "Line") +
  theme(
    #axis.text.x = element_text(angle = 30, hjust = 1),
    legend.position = "top",        # 保持在右侧，稍后会收集到panel外
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

ggsave(paste0(base_dir, "overall_metrics_barline.png"), width = 10, height = 6, dpi = 450)



