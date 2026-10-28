
library(purrr)
library(tidyverse)
library(stringr)

# --- Paths -------------------------------------------------------------------
# Run this script from its own directory. Paths are relative to it; override
# them with the environment variables named below.
processed_dir <- Sys.getenv("PROCESSED_DIR", "../../data/processed")
external_dir  <- Sys.getenv("EXTERNAL_DIR",  "../../data/external")
# -----------------------------------------------------------------------------

file_list <- list.files(path = file.path(external_dir, "gisaid_lineage_ids"), pattern = "*.csv")  # 获取当前目录下所有CSV文件的文件名
path_list <- file.path(external_dir, "gisaid_lineage_ids", file_list)
data_list <- map(path_list, read.csv, header = F,.progress = T)   # 使用map()函数逐个读取CSV文件并存储为数据框列表

# 使用 sub 函数去掉 .csv 后缀
filenames_no_csv <- sub("\\.csv$", "", file_list)
names(data_list) <- filenames_no_csv

# 使用 lapply 将每个data.frame中的name添加为一列
my_list_with_name <- lapply(names(data_list), function(lineage) {
  df <- data_list[[lineage]]
  df$lineage <- lineage  # 添加name列
  names(df)[1] <- "Accession_ID"
  return(df)
})

# 将list转换为单个data.frame
ID_lineage <- do.call(rbind, my_list_with_name) %>% distinct(Accession_ID,.keep_all = T)

spike_bind_escape_entry_updated <- read.csv(file.path(processed_dir, "spike_bind_escape_entry_updated.csv")) %>% 
  select(Accession_ID, label_mut, starts_with("spike_"))


df <- spike_bind_escape_entry_updated %>%
  mutate(
    lineage = str_extract(Accession_ID, "(?<=\\|).*")
  )

# 定义函数：判断突变编号
get_331_531_mut <- function(mut_str) {
  muts <- unlist(strsplit(mut_str, "\\|"))
  # 匹配突变编号，捕获3位数
  selected <- grep("^[A-Z][0-9]{3}[A-Z\\-]", muts, value = TRUE)
  # 提取数字部分，筛选在范围内
  selected <- selected[sapply(selected, function(x) {
    num <- as.numeric(str_extract(x, "[0-9]{3}"))
    !is.na(num) && num >= 331 && num <= 531
  })]
  paste(selected, collapse = "-")
}

df <- df %>%
  mutate(
    label_mut = sapply(label_mut, get_331_531_mut)
  )


data <-  read.csv(file.path(processed_dir, "all_data_with_SDI_updated.csv"))
data_filtered <-  data %>% filter(R >= 10 & R<= 200)

unique(data_filtered$lineage)

df_filtered <- df %>% filter(lineage %in% unique(data_filtered$lineage)) %>% 
  select(lineage, label_mut)

#write.csv(df_filtered,file.path(processed_dir, "label_mut_filtered.csv"),row.names = F)


##################################################
library(data.table)
library(purrr)

# 文件夹路径
folder_path <- file.path(external_dir, "msa")
file_list <- list.files(folder_path, pattern = "_msa\\.csv$", full.names = TRUE)

# 读取并处理：去第一行和第一列
read_skip_row_col <- function(file) {
  dt <- fread(file, header = T)
  return(dt[-1, -1])
}

# 合并
merged_data <- map_dfr(file_list, read_skip_row_col)

df_new <- cbind(merged_data, do.call(rbind, strsplit(merged_data$V1, "\\|")))
colnames(df_new)[ncol(df_new)-(2:0)] <- c("Accession_ID", "lineage", "date")

df_unique <- df_new %>%
  group_by(lineage) %>%
  arrange(date, .by_group = TRUE) %>%
  slice(1) %>%
  ungroup()

df_merged <- left_join(data_filtered, df_unique, by = "lineage", suffix = c("_filtered", "_unique")) %>% drop_na(Accession_ID)

write.csv(df_merged,file.path(processed_dir, "label_mut_seq_merged_updated.csv"),row.names = F)






