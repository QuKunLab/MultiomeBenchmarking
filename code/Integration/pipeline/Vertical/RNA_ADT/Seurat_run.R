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
DE_gene = read.csv(paste0(args[2],"/DEgene.csv"),header = TRUE)
M <- M[DE_gene[,2],]

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

ADT = read.table(file = paste0(args[1],"/ADT.csv"),header = 1,sep = ',',row.names = 1)
colnames(ADT) <- gsub('\\.','-',colnames(ADT))
ADT <- CreateAssayObject(counts = ADT)
pbmc[["ADT"]] <- ADT
rm(ADT)
DefaultAssay(pbmc) <- "ADT"
VariableFeatures(pbmc) <- rownames(pbmc[["ADT"]])
pbmc <- NormalizeData(pbmc, normalization.method = 'CLR', margin = 2) %>% 
  ScaleData() %>% RunPCA(reduction.name = 'apca')
dimension <- 20
if (dim(pbmc[["ADT"]])[1]-1 < 20) dimension <- dim(pbmc[["ADT"]])[1]-1
pbmc <- FindNeighbors(pbmc, dims = 1:dimension, reduction = "apca")
pbmc <- FindClusters(pbmc, resolution = 0.5)


pbmc <- FindMultiModalNeighbors(pbmc, reduction.list = list("pca", "apca"), dims.list = list(1:15, 1:dimension))

pbmc <- RunUMAP(pbmc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
pbmc <- FindClusters(pbmc, graph.name = "wsnn", algorithm = 3, verbose = FALSE)

writeMM(pbmc@graphs$wsnn,paste0(args[2], "/Seurat_connectivities.mtx"))
writeMM(pbmc@graphs$wknn,paste0(args[2], "/Seurat_distance.mtx"))