library(readr)
library(dplyr)
library(stringr)
library(purrr)

# --- Paths -------------------------------------------------------------------
# Run this script from its own directory. Paths are relative to it; override
# them with the environment variables named below.
processed_dir <- Sys.getenv("PROCESSED_DIR", "../../data/processed")
external_dir  <- Sys.getenv("EXTERNAL_DIR",  "../../data/external")
# -----------------------------------------------------------------------------


# =========================
# 1. 文件路径
# =========================
base_dir <- processed_dir

rate_files <- c(
  "Chile_rates.csv",
  "China_rates.csv",
  "Denmark_rates.csv",
  "Japan_rates.csv",
  "Mexico_rates.csv",
  "Nepal_rates.csv",
  "Poland_rates.csv"
)

spike_file <- file.path(base_dir, "spike_bind_escape_entry_updated.csv")
sdi_file   <- file.path(base_dir, "SDI_data.csv")
old_file   <- file.path(base_dir, "all_data_with_SDI.csv")

# =========================
# 2. 读取7个国家的 rates 数据
#    并加 country + 过滤 R
# =========================
rates_all <- map_dfr(rate_files, function(f) {
  df <- read_csv(file.path(base_dir, f), show_col_types = FALSE)
  
  country_name <- str_remove(f, "_rates\\.csv$")
  
  df %>%
    mutate(country = country_name) %>%
    filter(R >= 10, R <= 200) %>%
    select(lineage, R, country)
})

# =========================
# 3. 读取 spike 数据
#    从 Accession_ID 中提取 lineage
# =========================
spike_map <- read_csv(spike_file, show_col_types = FALSE) %>%
  mutate(
    lineage = str_trim(word(Accession_ID, -1, sep = fixed("|")))
  ) %>%
  select(
    lineage,
    spike_binding_score,
    spike_escape_score,
    spike_entry_score
  ) %>%
  distinct(lineage, .keep_all = TRUE)

# =========================
# 4. 读取 SDI 数据
# =========================
sdi_map <- read_csv(sdi_file, show_col_types = FALSE) %>%
  select(country, X2021) %>%
  rename(SDI = X2021)

# =========================
# 5. 合并：rates + spike + SDI
# =========================
new_7countries <- rates_all %>%
  left_join(spike_map, by = "lineage") %>%
  left_join(sdi_map, by = "country") %>%
  select(
    lineage,
    R,
    country,
    SDI,
    spike_binding_score,
    spike_escape_score,
    spike_entry_score
  )

# =========================
# 6. 检查有没有没匹配上的 lineage / country
# =========================
unmatched_lineage <- new_7countries %>%
  filter(is.na(spike_binding_score) | is.na(spike_escape_score) | is.na(spike_entry_score)) %>%
  distinct(lineage)

unmatched_country <- new_7countries %>%
  filter(is.na(SDI)) %>%
  distinct(country)

cat("新7国数据维度：", nrow(new_7countries), "x", ncol(new_7countries), "\n")
cat("未匹配上 spike score 的 lineage 数量：", nrow(unmatched_lineage), "\n")
cat("未匹配上 SDI 的 country 数量：", nrow(unmatched_country), "\n")

if (nrow(unmatched_lineage) > 0) {
  cat("前几个未匹配上的 lineage：\n")
  print(head(unmatched_lineage, 10))
}

if (nrow(unmatched_country) > 0) {
  cat("未匹配上的 country：\n")
  print(unmatched_country)
}

# =========================
# 7. 保存整理好的7国数据
#    这一步输出就是你截图那种格式
# =========================
write_csv(new_7countries, file.path(base_dir, "seven_countries_with_SDI_scores_filtered.csv"))

# =========================
# 8. 再与 all_data_with_SDI.csv 合并
# =========================
old_data <- read_csv(old_file, show_col_types = FALSE)

final_data <- bind_rows(old_data, new_7countries)
length(table(final_data$country))
# 如果你不想保留完全重复的行，可以打开这一句
# final_data <- distinct(final_data)

write_csv(final_data, file.path(base_dir, "all_data_with_SDI_updated.csv"))
mean(final_data$R)
median(final_data$R)
test_final_dat <- final_data[final_data$country %in% c('angladesh' , 'Brazil', 'Cambodia',  'Germany' ,  'India',  
                                     'Senegal',  'Turkey',  'UK' , 'USA'), "R"] 
mean(test_final_dat$R)
median(test_final_dat$R)

cat("文件已保存：\n")
cat("1) seven_countries_with_SDI_scores_filtered.csv\n")
cat("2) all_data_with_SDI_updated.csv\n")
