library(readr)
library(stringr)
library(tidyverse)

# --- Paths -------------------------------------------------------------------
# Run this script from its own directory. Paths are relative to it; override
# them with the environment variables named below.
processed_dir <- Sys.getenv("PROCESSED_DIR", "../../data/processed")
external_dir  <- Sys.getenv("EXTERNAL_DIR",  "../../data/external")
# -----------------------------------------------------------------------------


mydata <- read_tsv(file.path(external_dir, "cleaned_Kds_RBD_ACE2.tsv"))
df <- apply(do.call(rbind,str_split(mydata$geno, pattern = "") ),2 , as.numeric)
rowSums(apply(do.call(rbind,str_split(mydata$geno, pattern = "") ),2 , as.numeric))

five_order_binding_data <- mydata %>%
  filter(rowSums(apply(do.call(rbind,str_split(mydata$geno, pattern = "") ),2 , as.numeric)) <= 5 ) %>%
  mutate(deta_log10Kd = log10Kd_a - 8.937612) %>% select(geno, deta_log10Kd)

hist(five_order_binding_data$deta_log10Kd)