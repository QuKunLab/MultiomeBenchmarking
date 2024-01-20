setwd("/home/qukungroup/lyz32/Workspace/benchmark/code/ATAC_Real/deepmaps-master")
source("scRNA_scATAC1.r")
py_config()
# use_python("/home/math/hyl2016/miniconda3/envs/seuratV4_python38/bin/python3")
#Sys.setenv(RETICULATE_PYTHON = "/gpfs/home/math/hyl2016/miniconda3/envs/seuratV4_python38/bin/python")
#use_python("/gpfs/home/math/hyl2016/miniconda3/envs/seuratV4_python38/bin/python3")
#py_config()

args <- commandArgs(trailingOnly = TRUE)

# 检查输入文件是否存在
#if (!file.exists(args[1])) {
#  stop("Input data does not exist.")
#}

# 打印解析结果
#cat("Input file:", args[1], "\n")

path = paste0(args[1],'/ATAC/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
atac<-readMM(paste0(path,"matrix.mtx"))
row.names(atac) = Gene_name$V1
colnames(atac) = Cell_name$V1

path = paste0(args[1],'/RNA/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
rna<-readMM(paste0(path,"matrix.mtx"))
row.names(rna) = Gene_name$V1
colnames(rna) = Cell_name$V1

lymph_obj <- ReadData(rna_matrix = rna, atac_matrix = atac,data_type = "scRNA_scATAC", dataFormat = "matrixs", min_cell=0.005, nmad=3, gene.filter=TRUE, cell.filter=FALSE)

n <- nrow(lymph_obj@assays$ATAC@counts)
dia = Diagonal(n)
rownames(dia) <- rownames(lymph_obj@assays$ATAC@counts)
colnames(dia) <- 1:ncol(dia)
ATAC_gene_peak <-ATACCalculateGenescore(dia,
                         organism = "GRCm38",
                         decaydistance = 10000,
                         model = "Enhanced")
colnames(ATAC_gene_peak) <- rownames(lymph_obj@assays$ATAC@counts)
# ATAC_gene_peak <- CalGenePeakScore(peak_count_matrix = lymph_obj@assays$ATAC@counts,organism = "GRCh38")

GAS_obj <- calculate_GAS_v1(ATAC_gene_peak = ATAC_gene_peak, obj = lymph_obj, method = "wnn")
GAS <- GAS_obj[[1]]
lymph_obj <- GAS_obj[[2]]

HGT_result <- run_HGT(GAS = as.matrix(GAS),result_dir='./RNA_ATAC/', data_type='scRNA_scATAC', envPath=NULL, lr=0.2, epoch=30, n_hid=128, n_heads=16)
# write.table(HGTresult$cell_hgt_matrix, paste0(args[2],"/deepmaps.csv"), sep = ",", row.names = FALSE, col.names = FALSE)

cell_hgt_matrix <- HGT_result[['cell_hgt_matrix']]
rownames(cell_hgt_matrix) <- colnames(GAS)

lymph_obj <- lymph_obj[, colnames(GAS)]
cell_hgt_matrix <- cell_hgt_matrix[colnames(GAS),]

HGT_embedding <-
  CreateDimReducObject(embeddings = cell_hgt_matrix,
                       key = "HGT_",
                       assay = "RNA")

dir = strsplit(args[1],'/')[[1]]
write.table(HGT_embedding@cell.embeddings, paste0(paste0('/home/qukungroup/lyz32/Workspace/benchmark/code/ATAC_Real/DeepMaps/',dir[length(dir)]),'.csv'), sep = ",", row.names = FALSE, col.names = FALSE)


