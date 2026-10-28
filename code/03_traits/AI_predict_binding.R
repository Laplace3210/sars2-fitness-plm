# 定义 KD 值
data1 <- data.frame(
  Variant = c("Alpha (B.1.1.7)", "Delta (B.1.617.2)","Omicron (BA.1)", "Omicron (BA.2)", "Omicron (BA.3)"),
  KD = c(9.29451, 8.851085, 8.810884, 8.568085, 8.256491)-8.937612,
  SD = c(0, 0.2816464, 0.3805895, 0.3805895, 0.3805895) # 添加标准差（SD）((6.70^2/25.07^2)+(0.48^2/5.40^2))^0.5
)

# 计算 -log10(KD) 和 log10(SD)
data1$neg_log10_KD <- data1$KD
data1$log10_SD <- c(0, 0.2816464, 0.3805895, 0.3805895, 0.3805895)

# Prototype KD 值及其 -log10 计算
prototype_KD <- 0.356898
prototype_neg_log10_KD <- prototype_KD

# 计算 Δ(-log10KD)
data1$Delta_neg_log10_KD <- data1$neg_log10_KD - prototype_neg_log10_KD



data2 <- data.frame(
  Variant = c("Alpha (B.1.1.7)", "Delta (B.1.617.2)", "Omicron (BA.1)", "Omicron (BA.2)", "Omicron (BA.3)"),
  Delta_neg_log10_KD = c(0, 0.0362-0.230, -1.3655-0.230, -2.17-0.0366, -2.5-0.0366),
  SD = c(0, 0, 0,0,0)
)


# 合并两个数据集
data1$Source <- "Computed from AI-Based model predicted"
data2$Source <- "Computed from DMS data (Our study)"

combined_data <- rbind(data2[, c("Variant", "Delta_neg_log10_KD", "SD", "Source")],
                       data1[, c("Variant", "Delta_neg_log10_KD", "SD", "Source")])

# 自定义形状
shape_mapping <- c("Computed from AI-Based model predicted" = 17,  # 三角形
                   "Computed from DMS data (Our study)" = 16) 

# 让 data1 和 data2 共享颜色
color_mapping <- c("Computed from AI-Based model predicted" = "#A9767D",
                   "Computed from DMS data (Our study)" = scales::hue_pal()(3)[2])  

# 获取 data2 的颜色，用于右侧 Y 轴
data2_color <- color_mapping["Computed from DMS data (Our study)"]
data1_color <- color_mapping["Computed from AI-Based model predicted"]
# 绘制双 Y 轴的点图，并确保右侧 Y 轴颜色匹配 data3
library(ggplot2)
figure2E <- ggplot(combined_data, aes(x = Variant, y = Delta_neg_log10_KD, 
                                      color = Source, shape = Source, group = Source)) +
  geom_point(size = 4, position = position_dodge(width = 0.2)) +
  geom_line(position = position_dodge(width = 0.2)) +  # 添加组内点的连线
  # geom_errorbar(data = combined_data,  
  #               aes(ymin = Delta_neg_log10_KD - 1.96 * SD, 
  #                   ymax = Delta_neg_log10_KD + 1.96 * SD), 
  #               width = 0, position = position_dodge(width = 0.2)) +  # 只为 "Computed from Han et al.,2022" 添加误差条
  scale_y_continuous(
    name = "Binding to ACE2" # 右 Y 轴标签
  ) +
  scale_shape_manual(values = shape_mapping) +  # 指定点的形状
  scale_color_manual(values = color_mapping) +  # 指定颜色，data1 和 data2 共享红色
  #labs(title = "Comparison of VOC binding trend", x = "") +
  theme_bw() +
  theme(axis.text.x = element_text(size = 14), 
        axis.title.x = element_blank(),
        legend.position = c(0.8, 0.87),  # 图例右上方
        legend.background = element_blank(),  # 透明背景
        legend.text = element_text(size = 8),  # 调整图例文本大小
        legend.title = element_text(size = 10),  # 调整图例标题大小
        legend.key.size = unit(0.2, "cm"),  # 调整图例中点和线的大小
        axis.title.y = element_text(size = 16),  # 右侧 Y 轴标签颜色
        axis.text.y = element_text(size = 14)
  ) 
figure2E
