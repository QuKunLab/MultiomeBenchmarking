library(scAI)
library(dplyr)
library(cowplot)
library(ggplot2)
require(Matrix)

args <- commandArgs(trailingOnly = TRUE)

# 检查输入文件是否存在
if (!file.exists(args[1])) {
  stop("Input data does not exist.")
}

# 打印解析结果
cat("Input file:", args[1], "\n")

path = paste0(args[1],'/RNA/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = Cell_name$V1

path = paste0(args[1],'/ATAC/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
atac_counts<-readMM(paste0(path,"matrix.mtx"))
row.names(atac_counts) = Gene_name$V1
colnames(atac_counts) = Cell_name$V1


X <- list()
X$RNA <- M
X$ATAC <- atac_counts


start_time <- Sys.time()

scAI_outs <- create_scAIobject(raw.data = X)
scAI_outs <- preprocessing(scAI_outs, assay = list("RNA", "ATAC"), minFeatures = 1, minCells = 1)

scAI_outs <- run_scAI(scAI_outs, K = 20, nrun = 1, do.fast = T)
write.table(scAI_outs@fit$H, paste0(args[2],'scAI.csv'), sep = ",", row.names = FALSE, col.names = FALSE)

end_time <- Sys.time()
print(end_time - start_time)
