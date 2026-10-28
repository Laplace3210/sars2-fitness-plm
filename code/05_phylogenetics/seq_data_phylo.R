library(dplyr)
library(readr)

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


countries <- c(
  "Bangladesh", "Brazil", "Cambodia", "Germany", "India",
  "Senegal", "Turkey", "UK", "USA"
)

rds_dir <- file.path(external_dir, "raw_sequence_rds")
rates_dir <- file.path(external_dir, "MLR_fitting_R_res")
output_dir <- file.path(external_dir, "seq_data")

start_date <- as.Date("2021-11-01")
end_date <- as.Date("2022-11-30")

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

for (country in countries) {
  
  cat("Processing:", country, "\n")
  
  rds_file <- file.path(rds_dir, paste0(country, "_data.rds"))
  rates_file <- file.path(rates_dir, paste0(country, "_rates.csv"))
  
  seq_data <- readRDS(rds_file)
  rates <- read_csv(rates_file, show_col_types = FALSE)
  
  merged_data <- seq_data %>%
    left_join(rates, by = "lineage") %>%
    rename(
      sample_id = Accession_ID,
      collection_date = collection.date,
      sequence = seq.content,
      fitness = R
    ) %>%
    mutate(
      country = country,
      collection_date = as.Date(collection_date)
    )
  
  merged_data_fitness <- merged_data %>%
    filter(
      !is.na(fitness),
      collection_date >= start_date,
      collection_date <= end_date
    )
  
  metadata <- merged_data_fitness %>%
    select(sample_id, lineage, collection_date, country, fitness)
  
  write_csv(
    metadata,
    file.path(output_dir, paste0(country, "_metadata.csv"))
  )
  
  fasta_lines <- paste0(
    ">",
    merged_data_fitness$sample_id,
    "\n",
    merged_data_fitness$sequence
  )
  
  writeLines(
    fasta_lines,
    file.path(output_dir, paste0(country, "_spike_with_fitness.fasta"))
  )
  
  cat("Number of original sequences:", nrow(seq_data), "\n")
  cat("Number of sequences with fitness in date range:", nrow(merged_data_fitness), "\n\n")
}

cat("All countries finished.\n")