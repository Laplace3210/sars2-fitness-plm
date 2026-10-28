
library(tidyverse)
library(readr)

# --- Paths -------------------------------------------------------------------
# Run this script from its own directory. Paths are relative to it; override
# them with the environment variables named below.
processed_dir <- Sys.getenv("PROCESSED_DIR", "../../data/processed")
external_dir  <- Sys.getenv("EXTERNAL_DIR",  "../../data/external")
# -----------------------------------------------------------------------------

strain_fitness_data <- read_csv(file.path(external_dir, "df_existing_variants.csv")) %>% 
  select(mutations_x, average_date, log10Kd_ACE2, log10Kd_CB6,      
                       log10Kd_CoV555,log10Kd_REGN10987,log10Kd_S309,real_f,pred_f)

linea_fitness_data <- read_tsv(file.path(external_dir, "strains.tsv")) %>% 
  select(strain, `R / R_A`, mutation)

mut_lab_escape <- read_csv(file.path(processed_dir, "Antibody_binding_retained_result.csv"))

# 定义转换函数
mutations_to_string <- function(x) {
  # 去掉方括号和单引号
  x <- gsub("\\[|\\]|'", "", x)
  # 拆分成突变项
  muts <- unlist(strsplit(x, ",\\s*"))
  # 去掉S:前缀
  muts <- str_replace(muts, "S:", "")
  # 合并为字符串
  paste(muts, collapse = "-")
}

get_S_mutations_sorted <- function(mutstr, from = 331, to = 531) {
  all_muts <- unlist(strsplit(mutstr, ","))
  S_muts <- grep("^S:", all_muts, value = TRUE)
  extract_pos <- function(x) as.numeric(sub("^[^:]+:[A-Z](\\d+)[A-Z]$", "\\1", x))
  positions <- sapply(S_muts, extract_pos)
  in_range_idx <- which(!is.na(positions) & positions >= from & positions <= to)
  sorted_idx <- in_range_idx[order(positions[in_range_idx])]
  S_muts_sorted <- S_muts[sorted_idx]
  S_muts_sorted <- sub("^S:", "", S_muts_sorted)
  paste(S_muts_sorted, collapse = "-")
}

# mutstr <- "E:T9I,M:A63T,M:Q19E,N:G204R,N:G243S,N:P13L,N:R203K,N:S413R,ORF14:G50N,ORF1a:G1307S,ORF1a:L3027F,ORF1a:L3201F,ORF1a:P3395H,ORF1a:S135R,ORF1a:T3090I,ORF1a:T3255I,ORF1a:T842I,ORF1b:I1566V,ORF1b:P314L,ORF1b:R1315C,ORF1b:T2163I,ORF3a:T223I,ORF6:D61L,ORF9b:P10S,S:D405N,S:D614G,S:D796Y,S:E484A,S:G339D,S:H655Y,S:K417N,S:N440K,S:N501Y,S:N679K,S:N764K,S:N969K,S:P681H,S:Q493R,S:Q498R,S:Q954H,S:R408S,S:S371F,S:S373P,S:S375F,S:S477N,S:T19I,S:T376A,S:T478K,S:V213G,S:Y505H"
# 
# get_S_mutations_sorted(mutstr)
# 应用于整列
strain_fitness_data$label_mut <- sapply(strain_fitness_data$mutations_x, mutations_to_string)

linea_fitness_data$label_mut <- sapply(linea_fitness_data$mutation, get_S_mutations_sorted)

anti_all_data <- read.csv(file.path(processed_dir, "Antibody_binding_comb_all.csv"))
comb_dat <- strain_fitness_data %>% left_join(anti_all_data, by = 'label_mut') 




#write.csv(comb_dat$label_mut, file.path(processed_dir, "label_mut.csv"),row.names = F)
# 
# plot(comb_all$`R / R_A`,comb_all$pred_f)

write.csv(comb_dat, file.path(processed_dir, "comb_dat.csv"),row.names = F)

















