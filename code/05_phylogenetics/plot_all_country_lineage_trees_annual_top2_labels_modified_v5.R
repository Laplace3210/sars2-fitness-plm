library(ape)
library(readr)
library(dplyr)
library(ggplot2)
library(ggtree)
library(viridis)
library(scales)
library(jsonlite)
library(lubridate)
library(patchwork)
library(ggrepel)

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
# 1. Base directory
# ============================================================

base_dir <- tree_root

countries <- c(
  "Bangladesh", "Cambodia", "Senegal",
  "India", "Brazil", "Turkey",
  "USA", "UK", "Germany"
)

plot_mode <- "lineage_median"


# ============================================================
# Annual top fitness lineage labels
# ============================================================
# 每个国家、每个年度窗口内，自动标注 fitness 排名前 top_n 的 lineage。
# 例如 year_start = 2020 表示 2020-2021，即 x >= 2020 且 x < 2021。
annual_year_starts <- c(2020, 2021, 2022)
top_n_lineages_per_year <- 2

# 如果你担心某个 lineage 只有 1 条序列造成排名不稳定，可以改成 2 或 3。
min_samples_per_lineage_year <- 1

# ============================================================
# 2. Helper function: find metadata file
# ============================================================

find_metadata_file <- function(country_dir) {
  meta_tsv <- file.path(country_dir, "input", "metadata.tsv")
  meta_csv <- file.path(country_dir, "input", "metadata.csv")
  
  if (file.exists(meta_tsv)) {
    return(meta_tsv)
  } else if (file.exists(meta_csv)) {
    return(meta_csv)
  } else {
    stop("Cannot find metadata.tsv or metadata.csv in: ", file.path(country_dir, "input"))
  }
}

# ============================================================
# 3. Main plotting function for one country
# ============================================================

plot_one_country <- function(country,
                             base_dir,
                             save_individual = TRUE,
                             legend_inside = TRUE,
                             show_x_title = TRUE) {
  
  cat("\n============================================================\n")
  cat("Processing:", country, "\n")
  cat("============================================================\n")
  
  country_dir <- file.path(base_dir, country)
  
  tree_file <- file.path(country_dir, "results", "tree_time.nwk")
  node_data_file <- file.path(country_dir, "results", "node_data.json")
  metadata_file <- find_metadata_file(country_dir)
  
  output_dir <- file.path(country_dir, "updated_tree_image")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  if (!file.exists(tree_file)) {
    stop("Missing tree file: ", tree_file)
  }
  
  if (!file.exists(node_data_file)) {
    stop("Missing node_data.json: ", node_data_file)
  }
  
  # ============================================================
  # Read tree and node_data
  # ============================================================
  
  tree <- read.tree(tree_file)
  node_json <- fromJSON(node_data_file)
  
  clock_rate <- node_json$clock$rate
  root_decimal_year <- node_json$clock$rtt_Tmrca
  
  # root-to-tip regression rate from lookup table
  
  # annotation text shown at the top-left corner
  stat_text <- paste0(
    "Clock rate = ",
    formatC(clock_rate, format = "f", digits = 6)
  )
  cat("Tree tips:", length(tree$tip.label), "\n")
  cat("Clock rate:", clock_rate, "\n")
  cat("Root decimal year:", root_decimal_year, "\n")
  
  # ============================================================
  # Read metadata
  # ============================================================
  
  if (grepl("\\.tsv$", metadata_file)) {
    meta <- read_tsv(metadata_file, show_col_types = FALSE)
  } else {
    meta <- read_csv(metadata_file, show_col_types = FALSE)
  }
  
  required_cols <- c("strain", "lineage", "fitness")
  missing_cols <- setdiff(required_cols, colnames(meta))
  
  if (length(missing_cols) > 0) {
    stop(
      country,
      " metadata missing columns: ",
      paste(missing_cols, collapse = ", ")
    )
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
  # Calculate lineage-level fitness
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
  
  # -------- Percentile-clipped Min-Max scaling to 0–100 within each country --------
  # Clip extreme lineage fitness values at the 5th and 95th percentiles
  # before scaling to improve visualization contrast while preserving ranking.

  low_fit <- quantile(
    lineage_summary$fitness_median,
    0.05,
    na.rm = TRUE
  )

  high_fit <- quantile(
    lineage_summary$fitness_median,
    0.95,
    na.rm = TRUE
  )

  lineage_summary <- lineage_summary %>%
    mutate(
      fitness_median_clipped = pmin(
        pmax(fitness_median, low_fit),
        high_fit
      ),

      fitness_median_scaled =
        (fitness_median_clipped - low_fit) /
        (high_fit - low_fit) * 100,

      fitness_mean_scaled =
        (pmin(pmax(fitness_mean, low_fit), high_fit) - low_fit) /
        (high_fit - low_fit) * 100
    ) %>%
    arrange(desc(fitness_median_scaled))

  write_csv(
    lineage_summary,
    file.path(output_dir, paste0(country, "_lineage_fitness_summary_R.csv"))
  )
  
  # 如果没有 source 列，就用 fitness 是否为 NA 判断 global anchor
  if (!"source" %in% colnames(meta)) {
    meta$source <- ifelse(is.na(meta$fitness_num), "global_anchor", "country_data")
  }
  
  meta_plot <- meta %>%
    left_join(lineage_summary, by = "lineage") %>%
    mutate(
      fitness_plot = case_when(
        source == "global_anchor" ~ NA_real_,
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
  # Draw tree with ggtree
  # ============================================================
  
  p <- ggtree(
    tree,
    ladderize = TRUE,
    size = 0.22,
    color = "grey35"
  )
  
  max_x <- max(p$data$x, na.rm = TRUE)
  
  if (!is.null(clock_rate) && max_x < 0.1) {
    p$data$x <- root_decimal_year + p$data$x / clock_rate
  } else if (max_x < 10) {
    p$data$x <- root_decimal_year + p$data$x
  }
  
  # ============================================================
  # Metadata join
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
  
  # ============================================================
  # Annual top-2 lineage annotation data
  # ============================================================
  # 逻辑：
  # 1) 用时间树上的 x 坐标划分年度窗口：2020-2021, 2021-2022, 2022-2023。
  # 2) 在每个年度窗口内，按 lineage 汇总 sample-level fitness 的 median/mean。
  # 3) 每个年度窗口选择 fitness_median_year 排名前 top_n 的 lineage。
  # 4) 对每个被选中的 lineage，选一个最靠近年度窗口中点的 tip 放标签。
  #
  # Important: this only selects representative tips for labels.
  # It does NOT modify fitness_plot or the existing color mapping.
  
  plot_data_year <- plot_data %>%
    filter(
      isTip,
      !is.na(lineage),
      source != "global_anchor",
      !is.na(fitness_num),
      !is.na(x),
      x >= min(annual_year_starts),
      x < max(annual_year_starts) + 1
    ) %>%
    mutate(
      year_start = floor(x),
      year_end = year_start + 1,
      year_window = paste0(year_start, "-", year_end)
    ) %>%
    filter(year_start %in% annual_year_starts)
  
  annual_top_lineages <- plot_data_year %>%
    group_by(year_start, year_end, year_window, lineage) %>%
    summarise(
      n_samples_year = n(),
      fitness_median_year = median(fitness_num, na.rm = TRUE),
      fitness_mean_year = mean(fitness_num, na.rm = TRUE),
      fitness_min_year = min(fitness_num, na.rm = TRUE),
      fitness_max_year = max(fitness_num, na.rm = TRUE),
      # 这个值和图上颜色用的是同一个 0-100 country-level scaling，方便检查。
      fitness_plot_median = median(fitness_plot, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    filter(n_samples_year >= min_samples_per_lineage_year) %>%
    group_by(year_start, year_end, year_window) %>%
    arrange(desc(fitness_median_year), desc(n_samples_year), lineage, .by_group = TRUE) %>%
    mutate(annual_rank = row_number()) %>%
    filter(annual_rank <= top_n_lineages_per_year) %>%
    ungroup()
  
  label_tip_data <- plot_data_year %>%
    inner_join(
      annual_top_lineages,
      by = c("year_start", "year_end", "year_window", "lineage")
    ) %>%
    mutate(
      year_mid = year_start + 0.5,
      dist_to_year_mid = abs(x - year_mid),
      dist_to_year_median_fitness = abs(fitness_num - fitness_median_year),
      label_text = paste0(lineage, "\n", year_window)
    ) %>%
    group_by(year_start, year_window, lineage, annual_rank) %>%
    # 选最靠近年度窗口中点、且 fitness 更高的 tip 作为标签落点。
    arrange(dist_to_year_mid, desc(fitness_num), dist_to_year_median_fitness, .by_group = TRUE) %>%
    slice_head(n = 1) %>%
    ungroup()
  
  write_csv(
    annual_top_lineages,
    file.path(output_dir, paste0(country, "_annual_top2_lineage_summary.csv"))
  )
  
  write_csv(
    label_tip_data %>%
      select(
        label, lineage, year_window, annual_rank, x, y, source,
        fitness_num, fitness_median_year, fitness_mean_year,
        n_samples_year, fitness_plot, fitness_plot_median
      ),
    file.path(output_dir, paste0(country, "_annual_top2_lineage_label_points.csv"))
  )
  
  # 输出 tip-level mapping，方便后面检查
  tip_mapping <- plot_data %>%
    filter(isTip) %>%
    select(
      label,
      x,
      y,
      lineage,
      source,
      fitness_num,
      fitness_plot,
      fitness_median_scaled
    )
  
  write_csv(
    tip_mapping,
    file.path(output_dir, paste0(country, "_tip_fitness_mapping_R.csv"))
  )
  
  # ============================================================
  # Plot style
  # ============================================================
  
  fitness_title <- "Lineage Fitness\n(percentile-clipped scaled, 0–100)"
  plot_title <- country
  
  if (legend_inside) {
    legend_position <- c(0.05, 0.7)
    legend_theme <- theme(
      legend.position = legend_position,
      legend.justification = c(0, 0.5),
      legend.box = "vertical",
      legend.box.just = "left",
      legend.title = element_text(size = 15, hjust = 0),
      legend.text = element_text(size = 14, hjust = 0),
      legend.spacing.y = grid::unit(0.12, "cm"),
      legend.margin = margin(0, 0, 0, 0),
      legend.background = element_rect(
        fill = scales::alpha("white", 0.75),
        color = NA
      ),
      legend.box.background = element_blank()
    )
  } else {
    legend_theme <- theme(
      legend.position = "none"
    )
  }
  
  p_final <- p +
    geom_point(
      data = plot_data %>% filter(isTip, anchor_group == "Global anchor"),
      aes(x = x, y = y, shape = anchor_group),
      color = "grey75",
      size = 1.8,
      alpha = 0.9,
      inherit.aes = FALSE
    ) +
    geom_point(
      data = plot_data %>% filter(isTip, !is.na(fitness_plot)),
      aes(x = x, y = y, color = fitness_plot),
      size = 2,
      alpha = 0.95,
      inherit.aes = FALSE
    ) +
    # Red X markers for annual top-2 lineages.
    # Fixed color/shape outside aes(), so the original fitness color scale is unchanged.
    geom_point(
      data = label_tip_data,
      aes(x = x, y = y),
      shape = 4,
      color = "red",
      size = 4.8,
      stroke = 0.9,
      inherit.aes = FALSE,
      show.legend = FALSE
    ) +
    # Text labels for annual top-2 lineages
    ggrepel::geom_label_repel(
      data = label_tip_data,
      aes(x = x, y = y, label = label_text),
      inherit.aes = FALSE,
      size = 3.5,
      lineheight = 0.9,
      color = "black",
      fill = scales::alpha("white", 0.88),
      label.size = 0.25,
      box.padding = 0.28,
      point.padding = 0.18,
      min.segment.length = 0,
      segment.color = "grey35",
      segment.size = 0.25,
      seed = 123,
      max.overlaps = Inf,
      show.legend = FALSE
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
      name = if (show_x_title) "Inferred calendar time" else NULL,
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
        barwidth = grid::unit(6.2, "cm"),
        barheight = grid::unit(0.42, "cm")
      ),
      shape = guide_legend(
        order = 2,
        direction = "horizontal",
        override.aes = list(size = 2.5)
      )
    ) +
    theme_tree2() +
    theme(
      plot.title = element_text(
        face = "bold",
        size = 19,
        hjust = 0.5
      ),
      axis.title.x = element_text(size = 18),
      axis.text.x = element_text(size = 17),
      axis.ticks.y = element_blank(),
      axis.text.y = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    legend_theme +
    ggtitle(plot_title)+
    annotate(
      "label",
      x = 2020.02,
      y = max(plot_data$y, na.rm = TRUE) * 0.98,
      label = stat_text,
      hjust = 0,
      vjust = 1,
      size = 5,
      label.size = 0.25,
      fill = scales::alpha("white", 0.75),
      color = "black"
    )
  
  # ============================================================
  # Save individual outputs
  # ============================================================
  
  if (save_individual) {
    out_prefix <- file.path(
      output_dir,
      paste0(country, "_timetree_lineage_median_fitness_R_annual_top2_labels")
    )
    
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
      dpi = 350
    )
    
    cat("Saved:", paste0(out_prefix, ".pdf"), "\n")
    cat("Saved:", paste0(out_prefix, ".png"), "\n")
  }
  
  return(p_final)
}

# ============================================================
# 4. Run all countries and save individual plots
# ============================================================

individual_plots <- list()

for (country in countries) {
  individual_plots[[country]] <- plot_one_country(
    country = country,
    base_dir = base_dir,
    save_individual = TRUE,
    legend_inside = TRUE,
    show_x_title = country %in% c("USA", "UK", "Germany")
  )
}

# ============================================================
# 5. Build 3 × 3 combined figure
# ============================================================

plot_order <- c(
  "Bangladesh", "Cambodia", "Senegal",
  "India", "Brazil", "Turkey",
  "USA", "UK", "Germany"
)

grid_plots <- list()

for (country in plot_order) {
  
  if (country == "Bangladesh") {
    # Bangladesh (Panel A) 保留图例
    grid_plots[[country]] <- individual_plots[[country]] +
      theme(
        plot.title = element_text(
          face = "bold",
          size = 19,
          hjust = 0.5
        ),
        axis.title.x = element_text(size = 18),
        axis.text.x = element_text(size = 17),
        
        # 保留 Germany 图里的图例
        legend.position = c(0.05, 0.7),
        legend.justification = c(0, 0.5),
        legend.box = "vertical",
        legend.box.just = "left",
        
        legend.title = element_text(size = 16, hjust = 0),
        legend.text = element_text(size = 15, hjust = 0),
        
        legend.spacing.y = grid::unit(0.12, "cm"),
        legend.margin = margin(0, 0, 0, 0),
        
        legend.background = element_rect(
          fill = scales::alpha("white", 0.75),
          color = NA
        ),
        legend.box.background = element_blank()
      )
    
  } else {
    # 其他国家不显示图例
    grid_plots[[country]] <- individual_plots[[country]] +
      theme(
        legend.position = "none",
        plot.title = element_text(
          face = "bold",
          size = 18,
          hjust = 0.5
        ),
        axis.title.x = element_text(size = 18),
        axis.text.x = element_text(size = 17)
      )
  }
}

combined_plot <- wrap_plots(
  grid_plots[plot_order],
  ncol = 3,
  byrow = TRUE
) +
  plot_annotation(tag_levels = "A") &
  theme(
    plot.tag = element_text(
      size = 30,
      face = "bold"
    ),
    plot.tag.position = c(0.02, 0.98)
  )
combined_output_dir <- file.path(base_dir, "combined_3x3")
dir.create(combined_output_dir, recursive = TRUE, showWarnings = FALSE)

ggsave(
  file.path(combined_output_dir, "All_countries_timetree_3x3_annual_top2_labels.pdf"),
  combined_plot,
  width = 22,
  height = 16,
  units = "in"
)

ggsave(
  file.path(combined_output_dir, "All_countries_timetree_3x3_annual_top2_labels.png"),
  combined_plot,
  width = 22,
  height = 16,
  units = "in",
  dpi = 650
)

cat("\n============================================================\n")
cat("All done.\n")
cat("Combined figure saved to:\n")
cat(file.path(combined_output_dir, "All_countries_timetree_3x3_annual_top2_labels.pdf"), "\n")
cat(file.path(combined_output_dir, "All_countries_timetree_3x3_annual_top2_labels.png"), "\n")
cat("============================================================\n")