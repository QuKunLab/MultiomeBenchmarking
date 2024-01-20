library(Seurat)
require(Matrix)
library(dplyr) 
require(stringr)
require(Signac)
# require(bindSC)
library(EnsDb.Hsapiens.v86)

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

pbmc <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 1, min.features = 1)
rm(M)
DefaultAssay(pbmc) <- "RNA"
pbmc <- NormalizeData(pbmc)
pbmc <- FindVariableFeatures(pbmc, nfeatures = 4000)
pbmc <- ScaleData(pbmc)
pbmc <- RunPCA(pbmc)
pbmc <- FindNeighbors(pbmc, dims = 1:20, reduction = "pca")
pbmc <- FindClusters(pbmc, resolution = 0.5)
pbmc <- RunUMAP(pbmc, reduction = "pca", dims = 1:15)

path = paste0(args[1],'/ATAC/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
atac_counts<-readMM(paste0(path,"matrix.mtx"))
rownames(atac_counts) = Gene_name$V1
colnames(atac_counts) = Cell_name$V1

chrom_assay <- CreateChromatinAssay(
counts = atac_counts,
sep = c("-", "-"),
# genome = 'hg38',
fragments = NULL
# min.cells = 10
# annotation = annotations
)
pbmc[["ATAC"]] <- chrom_assay
DefaultAssay(pbmc) <- "ATAC"
pbmc <- RunTFIDF(pbmc)
pbmc <- FindTopFeatures(pbmc, min.cutoff = 'q0')
pbmc <- RunSVD(pbmc,n = 50)
pbmc <- FindNeighbors(pbmc, dims = 1:20, reduction = "lsi")
pbmc <- FindClusters(pbmc, resolution = 0.5)
pbmc <- RunUMAP(pbmc, reduction = "lsi", dims = 1:15)

pbmc <- FindMultiModalNeighbors(pbmc, reduction.list = list("pca", "lsi"), dims.list = list(1:15, 1:15))

pbmc <- RunUMAP(pbmc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
pbmc <- FindClusters(pbmc, graph.name = "wsnn", algorithm = 3, verbose = FALSE)

dir <- strsplit(args[1],'/')[[1]]
writeMM(pbmc@graphs$wsnn,paste0(args[2],"Seurat_connectivities.mtx"))
writeMM(pbmc@graphs$wknn,paste0(args[2],"Seurat_distance.mtx"))
