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
                             legend_inside = TRUE) {
  
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
  
  # -------- Min-Max scaling to 0–100 within each country --------
  min_fit <- min(lineage_summary$fitness_median, na.rm = TRUE)
  max_fit <- max(lineage_summary$fitness_median, na.rm = TRUE)
  
  lineage_summary <- lineage_summary %>%
    mutate(
      fitness_median_scaled = (fitness_median - min_fit) / (max_fit - min_fit) * 100,
      fitness_mean_scaled   = (fitness_mean   - min_fit) / (max_fit - min_fit) * 100
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
  
  fitness_title <- "Lineage Fitness\n(min–max scaled, 0–100)"
  plot_title <- country
  
  if (legend_inside) {
    legend_position <- c(0.05, 0.75)
    legend_theme <- theme(
      legend.position = legend_position,
      legend.justification = c(0, 0.5),
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
    ggtitle(plot_title)
  
  # ============================================================
  # Save individual outputs
  # ============================================================
  
  if (save_individual) {
    out_prefix <- file.path(
      output_dir,
      paste0(country, "_timetree_lineage_median_fitness_R")
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
      dpi = 300
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
    legend_inside = TRUE
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
  
  if (country == "Germany") {
    # Germany 保留图例，并使用你原来设定的图例位置
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
        legend.position = c(0.05, 0.75),
        legend.justification = c(0, 0.5),
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
        axis.text.x = element_text(size = 17),
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
  file.path(combined_output_dir, "All_countries_timetree_3x3.pdf"),
  combined_plot,
  width = 22,
  height = 16,
  units = "in"
)

ggsave(
  file.path(combined_output_dir, "All_countries_timetree_3x3.png"),
  combined_plot,
  width = 22,
  height = 16,
  units = "in",
  dpi = 350
)

cat("\n============================================================\n")
cat("All done.\n")
cat("Combined figure saved to:\n")
cat(file.path(combined_output_dir, "All_countries_timetree_3x3.pdf"), "\n")
cat(file.path(combined_output_dir, "All_countries_timetree_3x3.png"), "\n")
cat("============================================================\n")