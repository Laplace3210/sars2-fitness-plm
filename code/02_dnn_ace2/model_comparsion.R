

library(ggplot2)
library(openxlsx)
library(dplyr)
library(readr)

# --- Paths -------------------------------------------------------------------
# Run this script from its own directory (code/02_dnn_ace2/).
base_dir <- "../../results/model_comparison"
fig_dir  <- "../../results/figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
# -----------------------------------------------------------------------------

# 读取数据
cnn <- read.xlsx(file.path(base_dir, "CNN.xlsx")) %>% mutate(epoch = row_number(), model = "CNN")
lstm <- read.xlsx(file.path(base_dir, "LSTM.xlsx")) %>% mutate(epoch = row_number(), model = "LSTM")
mlp <- read.xlsx(file.path(base_dir, "MLP.xlsx")) %>% mutate(epoch = row_number(), model = "MLP")
rnn <- read.xlsx(file.path(base_dir, "RNN.xlsx")) %>% mutate(epoch = row_number(), model = "RNN")

# 合并数据
all_data <- bind_rows(cnn, lstm, mlp, rnn)
model_colors <- c("CNN" = "#66c2a5", "LSTM" = "#fc8d62", "MLP" = "#8da0cb", "RNN" = "#e78ac3")
# 绘图
p_loss <- ggplot(all_data, aes(x = epoch, y = value, color = model)) +
  geom_line(size = 1) +
  scale_color_manual(values = model_colors)+
  theme_bw(base_size = 14) +
  labs(
    x = "Epoch",
    y = "Loss",
    color = "Model"
  ) +
  guides(color = guide_legend(ncol = 2)) +
  theme(
    #plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.text.x = element_text(size=12),
    axis.text.y = element_text(size = 12),
    legend.position = c(0.95, 0.99),  # 图右上角
    legend.justification = c("right", "top"),
    legend.background = element_blank(),  # 移除背景
    legend.box.background = element_blank(),  # 移除外框
    legend.title = element_text(size = 10,face = "bold"),
    legend.text = element_text(size = 8), 
    #panel.grid.major = element_line(color = "gray80", linetype = "dashed"),
    panel.grid.minor = element_blank()
  )



# 读取每个模型的结果文件（第一列为真实值，第二列为预测值）
cnn <- read_csv(file.path(base_dir, "CNN_results.csv"), col_names = T)
lstm <- read_csv(file.path(base_dir, "LSTM_results.csv"), col_names = T)
mlp <- read_csv(file.path(base_dir, "MLP_results.csv"), col_names = T)
rnn <- read_csv(file.path(base_dir, "RNN_results.csv"), col_names = T)

# 定义函数：计算 MSE 和 PCC
get_metrics <- function(df, model_name) {
  y_true <- as.numeric(df[[1]])
  y_pred <- as.numeric(df[[2]])
  mse <- mean((y_true - y_pred)^2, na.rm = TRUE)
  pcc <- cor(y_true, y_pred, use = "complete.obs")
  return(data.frame(Model = model_name, MSE = mse, PCC = pcc))
}

# 计算所有模型的指标
results <- bind_rows(
  get_metrics(cnn, "CNN"),
  get_metrics(lstm, "LSTM"),
  get_metrics(mlp, "MLP"),
  get_metrics(rnn, "RNN")
)




# 保留原始颜色映射顺序
#model_colors <- c("CNN" = "#66c2a5", "LSTM" = "#fc8d62", "MLP" = "#8da0cb", "RNN" = "#e78ac3")

# 将模型按 MSE 升序排列，但颜色保持一致
results$Model <- factor(results$Model, levels = results$Model[order(results$MSE)])

# 绘图
p_MSE <- ggplot(results, aes(x = Model, y = MSE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  scale_fill_manual(values = model_colors) +
  theme_bw(base_size = 14) +
  labs(
    x = "Model",
    y = "MSE"
  ) +
  theme( axis.text.x = element_text(size=12),
            axis.text.y = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "none"
  )



# 读取数据（第一列为真实值，第二列为预测值）
mlp <- read_csv(file.path(base_dir, "MLP_results.csv"), 
                col_names = T) 
colnames(mlp) <- c("True", "Predicted")

# 绘制散点图
p_scatter <-  ggplot(mlp, aes(x = Predicted, y = True)) +
  geom_point(alpha = 0.6, color = "#8da0cb", size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 1) +
  scale_x_continuous(breaks = seq(8, 10, by = 0.5), limits = c(8, 10)) +  # ✅ 指定刻度与范围
  theme_bw(base_size = 14) +
  labs(x = "MLP model predicted", y = "Observed experimental data\n (ACE2 binding)") +
  theme(
    axis.text.x = element_text(size = 14),
    axis.text.y = element_text(size = 14)
    # panel.grid.major = element_line(color = "gray80", linetype = "dashed"),
    # panel.grid.minor = element_blank()
  )+
  annotate("text", x = 10, y = 8.5, label = expression(rho == 0.84), 
           hjust = 1.1, vjust = 1.5, size = 5, fontface = "bold")


library(patchwork)
# ==== 拼图：p1 占一整行，p2 和 p3 并列 ====
combined_plot <- p_loss / (p_MSE | p_scatter) +
  plot_annotation(tag_levels = 'A')  # 自动标 A, B, C

# 显示图像
combined_plot

ggsave(combined_plot, file = file.path(fig_dir, "model_comparison.png"), width=8, height=6.5, dpi = 400)

