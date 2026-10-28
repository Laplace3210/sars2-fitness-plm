
library(ape)
library(readr)
library(dplyr)
library(ggplot2)
library(ggtree)
library(viridis)
library(scales)
library(jsonlite)
library(lubridate)

# --- Paths -------------------------------------------------------------------
# Run this script from its own directory. The time-resolved trees and the raw
# sequence data are not tracked in this repository (they are rebuilt from GISAID
# records); set the variables below, or the matching environment variables, to
# wherever you keep them.
tree_root     <- Sys.getenv("TREE_RESULTS", "../../data/external/tree_results")
external_dir  <- Sys.getenv("EXTERNAL_DIR", "../../data/external")
fig_dir       <- Sys.getenv("FIGURE_DIR",   "../../results/figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
# -----------------------------------------------------------------------------


# ============================================================
# 1. Set working directory
# ============================================================

#setwd(file.path(tree_root, "UK", "results"))

tree_file <- file.path(tree_root, "UK", "results", "tree_time.nwk")
node_data_file <- file.path(tree_root, "UK", "results", "node_data.json")

# 如果你是 metadata.csv，就改成 metadata.csv
metadata_file <- file.path(tree_root, "UK", "input", "metadata.tsv")
# "lineage_median" 推荐用于映射 lineage-level fitness

# "sample" 表示每个 tip 使用自己的 fitness
plot_mode <- "lineage_median"
# plot_mode <- "sample"

# ============================================================
# 2. Read tree and node_data
# ============================================================

tree <- read.tree(tree_file)
node_json <- fromJSON(node_data_file)

clock_rate <- node_json$clock$rate
root_decimal_year <- node_json$clock$rtt_Tmrca

cat("Tree tips:", length(tree$tip.label), "\n")
cat("Clock rate:", clock_rate, "\n")
cat("Root decimal year:", root_decimal_year, "\n")

# ============================================================
# 3. Read metadata
# ============================================================

if (grepl("\\.tsv$", metadata_file)) {
  meta <- read_tsv(metadata_file, show_col_types = FALSE)
} else {
  meta <- read_csv(metadata_file, show_col_types = FALSE)
}

required_cols <- c("strain", "lineage", "fitness")
missing_cols <- setdiff(required_cols, colnames(meta))

if (length(missing_cols) > 0) {
  stop(paste("metadata missing columns:", paste(missing_cols, collapse = ", ")))
}

meta <- meta %>%
  mutate(
    fitness_num = suppressWarnings(as.numeric(fitness))
  )

cat("Metadata rows:", nrow(meta), "\n")
cat("Numeric fitness rows:", sum(!is.na(meta$fitness_num)), "\n")
cat("Tree tips missing in metadata:", length(setdiff(tree$tip.label, meta$strain)), "\n")
cat("Metadata rows not in tree:", length(setdiff(meta$strain, tree$tip.label)), "\n")

# ============================================================
# 4. Calculate lineage-level fitness
# ============================================================

lineage_summary <- meta %>%
  filter(!is.na(fitness_num)) %>%
  group_by(lineage) %>%
  summarise(
    n_samples = n(),
    fitness_mean = mean(fitness_num, na.rm = TRUE),
    fitness_median = median(fitness_num, na.rm = TRUE),
    fitness_min = min(fitness_num, na.rm = TRUE),
    fitness_max = max(fitness_num, na.rm = TRUE),
    .groups = "drop"
  )

# -------- Min-Max scaling to 0–100 --------
min_fit <- min(lineage_summary$fitness_median, na.rm = TRUE)
max_fit <- max(lineage_summary$fitness_median, na.rm = TRUE)

lineage_summary <- lineage_summary %>%
  mutate(
    fitness_median_scaled = (fitness_median - min_fit) / (max_fit - min_fit) * 100,
    fitness_mean_scaled   = (fitness_mean   - min_fit) / (max_fit - min_fit) * 100
  ) %>%
  arrange(desc(fitness_median_scaled))

write_csv(lineage_summary, "lineage_fitness_summary_R.csv")

meta_plot <- meta %>%
  left_join(lineage_summary, by = "lineage") %>%
  mutate(
    fitness_plot = case_when(
      plot_mode == "lineage_median" ~ fitness_median_scaled,
      plot_mode == "sample" ~ fitness_num,
      TRUE ~ fitness_median_scaled
    ),
    anchor_group = case_when(
      source == "global_anchor" ~ "Global anchor",
      TRUE ~ NA_character_
    )
  )
# ============================================================
# 5. Draw tree with ggtree
# ============================================================

p <- ggtree(
  tree,
  ladderize = TRUE,
  size = 0.22,
  color = "grey35"
)

# ------------------------------------------------------------
# Convert branch length to calendar year
# Augur tree_time.nwk 的 branch length 通常是 clock length，
# 需要用 node_data.json 里的 clock rate 转成年份。
# ------------------------------------------------------------

max_x <- max(p$data$x, na.rm = TRUE)

if (!is.null(clock_rate) && max_x < 0.1) {
  p$data$x <- root_decimal_year + p$data$x / clock_rate
} else if (max_x < 10) {
  p$data$x <- root_decimal_year + p$data$x
}

# ============================================================
# 6. Color scale
# ============================================================

fitness_limits <- quantile(
  meta_plot$fitness_plot,
  probs = c(0.01, 0.99),
  na.rm = TRUE
)

fitness_title <- ifelse(
  plot_mode == "lineage_median",
  "Lineage Fitness\n(min–max scaled, 0–100)",
  "Sample Fitness"
)

plot_title <- ifelse(
  plot_mode == "lineage_median",
  "UK"#,
  #"UK timetree colored by sample-level fitness"
)

# ============================================================
# 7. Final plot
# ============================================================

tip_annotation <- meta_plot %>%
  select(
    strain,
    lineage,
    source,
    anchor_group,
    fitness_num,
    fitness_plot,
    fitness_median,
    fitness_mean,
    fitness_median_scaled,
    fitness_mean_scaled,
    n_samples
  )

plot_data <- p$data %>%
  left_join(
    tip_annotation,
    by = c("label" = "strain")
  )

# 拆成两层：
# 1) Global anchor：灰色点，并单独出图例
# 2) Numeric fitness：连续颜色条

p_final <- p +
  # Global anchor layer
  geom_point(
    data = plot_data %>% filter(isTip, anchor_group == "Global anchor"),
    aes(x = x, y = y, shape = anchor_group),
    color = "grey75",
    size = 1.3,
    alpha = 0.9,
    inherit.aes = FALSE
  ) +
  # Fitness layer
  geom_point(
    data = plot_data %>% filter(isTip, !is.na(fitness_plot)),
    aes(x = x, y = y, color = fitness_plot),
    size = 1.3,
    alpha = 0.95,
    inherit.aes = FALSE
  ) +
  scale_shape_manual(
    name = NULL,
    values = c("Global anchor" = 16)
  ) +
  scale_color_viridis_c(
    name = fitness_title,
    option = "viridis",
    limits = c(0, 100),
    breaks = c(0, 25, 50, 75, 100),
    oob = scales::squish,
    na.value = "grey78"
  ) +
  scale_x_continuous(
    name = "Inferred calendar time",
    breaks = seq(2020, 2023, by = 1),
    limits = c(2019.9, 2023.1),
    expand = c(0.01, 0.01)
  ) +
  guides(
    color = guide_colorbar(
      order = 1,
      direction = "horizontal",
      title.position = "top",
      title.hjust = 0,
      label.position = "bottom",
      barwidth = grid::unit(5.5, "cm"),
      barheight = grid::unit(0.35, "cm")
    ),
    shape = guide_legend(
      order = 2,
      direction = "horizontal",
      override.aes = list(size = 2.5)
    )
  ) + theme_tree2() +
  theme(
    plot.title = element_text(face = "bold", size = 15,hjust = 0.5),
    #plot.subtitle = element_text(size = 10),
    axis.title.x = element_text(size = 11),
    axis.text.x = element_text(size = 10),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    
    # 图例放在图内
    legend.position = c(0.05, 0.75),
    legend.justification = c(0, 0.5),
    
    # 多个图例纵向排列，并且左对齐
    legend.box = "vertical",
    legend.box.just = "left",
    
    legend.title = element_text(size = 10, hjust = 0),
    legend.text = element_text(size = 9, hjust = 0),
    
    legend.spacing.y = grid::unit(0.12, "cm"),
    legend.margin = margin(0, 0, 0, 0),
    
    legend.background = element_rect(
      fill = scales::alpha("white", 0.75),
      color = NA
    ),
    legend.box.background = element_blank(),
    
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )+
  ggtitle(
    plot_title#,
    # subtitle = paste0(
    #   "tips = ", length(tree$tip.label),
    #   " | numeric fitness = ", sum(!is.na(meta$fitness_num)),
    #   " | global anchors = ", sum(meta$source == "global_anchor", na.rm = TRUE)
    # )
  )

print(p_final)

# ============================================================
# 8. Save output
# ============================================================

if (plot_mode == "lineage_median") {
  out_prefix <- file.path(fig_dir, "UK_timetree_lineage_median_fitness_R")
} else {
  out_prefix <- file.path(fig_dir, "UK_timetree_sample_fitness_R")
}

ggsave(
  paste0(out_prefix, ".pdf"),
  p_final,
  width = 6.19,
  height = 4.46,
  units = "in"
)

ggsave(
  paste0(out_prefix, ".png"),
  p_final,
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

cat("Saved:", paste0(out_prefix, ".pdf"), "\n")
cat("Saved:", paste0(out_prefix, ".png"), "\n")
cat("Saved: lineage_fitness_summary_R.csv\n")