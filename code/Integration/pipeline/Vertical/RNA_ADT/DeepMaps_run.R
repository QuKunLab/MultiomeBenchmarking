# library packages
library(Seurat)
library(plyr)
library(dsb)
library(ComplexHeatmap)
library(RColorBrewer)
library(reticulate)
library(CellChat)
library(patchwork)
library(scater)
library(Matrix)
source('/home/qukungroup/lyz32/Workspace/benchmark/code/RealData/DeepMaps_sh/citefunction.R')

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
rna<-readMM(paste0(path,"matrix.mtx"))
row.names(rna) = Gene_name$V1
colnames(rna) = Cell_name$V1
DE_gene = read.csv(paste0(args[2],"/DEgene.csv"),header = TRUE)
rna <- rna[DE_gene[,2],]

adt = read.table(file = paste0(args[1],"/ADT.csv"),header = 1,sep = ',',row.names = 1)
adt = as.matrix(adt)
colnames(adt) = colnames(rna)

PBMCandLung_obj <- ReadData(rna_matrix = rna, adt_matrix = adt, data_type = 'CITE', dataFormat='matrixs', gene.filter=FALSE, cell.filter=FALSE)
GAS <- CLR(obj = PBMCandLung_obj)
rm(rna)
rm(adt)

#envPath = "/home/math/hyl2016/miniconda3/envs/seuratV4_python38"
result_dir <- 'cite'
HGTresult <- run_HGT(GAS=GAS, result_dir=result_dir, data_type='CITE', lr=0.08, epoch=100, n_hid=128, n_heads=8,cuda=0)

write.table(HGTresult$cell_hgt_matrix, file = paste0(args[2], "/DeepMaps.csv"), sep = ",",
            row.names = FALSE,col.names = F)