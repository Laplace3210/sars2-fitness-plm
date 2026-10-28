library(tidyverse)
library(openxlsx)

# --- Paths -------------------------------------------------------------------
# Run this script from its own directory. Paths are relative to it; override
# them with the environment variables named below.
external_dir <- Sys.getenv("EXTERNAL_DIR", "../../data/external")
# -----------------------------------------------------------------------------

wild_type <- 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

# Number of mutations relative to the wild-type RBD (Hamming distance)
count_mutations <- function(seq1, seq2) {
  sum(strsplit(seq1, "")[[1]] != strsplit(seq2, "")[[1]])
}

# Keep only variants carrying 1-5 mutations
filter_five_loci <- function(df) {
  df$mutation_count <- sapply(df$mutant_sequence,
                              function(x) count_mutations(x, wild_type))
  subset(df, mutation_count <= 5) %>% select(-mutation_count)
}

# =============================================================================
# ESM-1v -- the analysis set used in the paper (4,923 measurements)
# =============================================================================
raw_data_15_mut <- read.csv(file.path(external_dir, "df_Desai_15loci_complete.csv"))
filtered_df <- filter_five_loci(raw_data_15_mut)
write.csv(filtered_df, file.path(external_dir, "df_Desai_5loci.csv"))

# =============================================================================
# ESM-2 -- exploratory only; NOT used in any published result. Kept so that the
# embedding comparison reported in the Supplementary Methods can be repeated.
# =============================================================================
raw_data_15_mut <- read.csv(file.path(external_dir, "df_15loci_with_mean_representation_esm2.csv"))
filtered_df <- filter_five_loci(raw_data_15_mut)
write.csv(filtered_df, file.path(external_dir, "df_esm2_5loci.csv"))
