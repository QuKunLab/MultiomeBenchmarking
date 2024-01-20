library(Seurat)
library(ggplot2)
library(Matrix)
library(Signac)
library(stats4)
library(dplyr)
library(Matrix)
library(SeuratDisk)
save_path <- './Results/'
data_dir <-  './Dataset37/'
TRAIN_RNAfile = paste0(data_dir,"Dataset37_TRAIN_rna.h5")
TRAIN_RNA <- Read10X_h5(TRAIN_RNAfile)
TRAIN_ATACfile = paste0(data_dir,"Dataset37_TRAIN_atac.h5")
TRAIN_ATAC <- Read10X_h5(TRAIN_ATACfile)
TEST_RNAfile = paste0(data_dir,"Dataset37_TEST_rna.h5")
TEST_RNA <- Read10X_h5(TEST_RNAfile)
TEST_ATACfile = paste0(data_dir,"Dataset37_TEST_atac.h5")
TEST_ATAC <- Read10X_h5(TEST_ATACfile)
TRAIN_RNA_obj <- CreateSeuratObject(counts = TRAIN_RNA, assay="RNA",project = "RNA", min.cells = 0, min.features = 0)
TRAIN_RNA_obj <- NormalizeData(TRAIN_RNA_obj, normalization.method = "LogNormalize", scale.factor = 1e4) 
TRAIN_RNA_obj <- FindVariableFeatures(TRAIN_RNA_obj, selection.method = 'vst', nfeatures = 4000)
TEST_RNA_obj <- CreateSeuratObject(counts = TEST_RNA, assay="RNA",project = "RNA", min.cells = 1, min.features = 0)
TEST_RNA_obj <- NormalizeData(TEST_RNA_obj, normalization.method = "LogNormalize", scale.factor = 1e4) 
ATAC_chromassay <- CreateChromatinAssay(counts = TRAIN_ATAC,sep = c("-", "-"),min.cells = 0,min.features = 0)
TRAIN_ATAC_obj <- CreateSeuratObject(counts = ATAC_chromassay, assay="ATAC",project = "ATAC", min.cells = 0, min.features = 0)
ATAC_chromassay <- CreateChromatinAssay(counts = TEST_ATAC,sep = c("-", "-"),min.cells = 0,min.features = 0)
TEST_ATAC_obj <- CreateSeuratObject(counts = ATAC_chromassay, assay="ATAC",project = "ATAC", min.cells = 0, min.features = 0)
features<-TRAIN_RNA_obj@assays$RNA@var.features
TRAIN_RNA <- TRAIN_RNA_obj@assays$RNA@counts
TRAIN_ATAC <- TRAIN_ATAC_obj@assays$ATAC@counts
TRAIN_rna_combine <- rbind(TRAIN_RNA,as(as.matrix(TRAIN_ATAC), "dgCMatrix"))
Ligerex.leaveout <- createLiger(list(train = TRAIN_rna_combine,test = TEST_RNA))
Ligerex.leaveout <- rliger::normalize(Ligerex.leaveout)
Ligerex.leaveout@norm.data <- Ligerex.leaveout@raw.data
Ligerex.leaveout <- selectGenes(Ligerex.leaveout)
Ligerex.leaveout <- scaleNotCenter(Ligerex.leaveout)
k = (length(Ligerex.leaveout@var.genes)-3)
if (k>20){
k = 20
}
options (warn = -1)
Ligerex.leaveout <- optimizeALS(Ligerex.leaveout,k = k, lambda = 20)
Ligerex.leaveout <- quantile_norm(Ligerex.leaveout)
options (warn = -1)
Ligerex.leaveout@norm.data$train <- rbind(TRAIN_rna_combine,as(as.matrix(TRAIN_ATAC), "dgCMatrix"))
Ligerex.leaveout@raw.data$train <- rbind(TRAIN_rna_combine,as(as.matrix(TRAIN_ATAC), "dgCMatrix"))
Imputation <- imputeKNN(Ligerex.leaveout,reference = 'train', queries = list('test'), norm = FALSE, scale = FALSE, knn_k = 40)##20,10
Result = as.matrix(Imputation@raw.data$test)[unlist(row.names(TRAIN_ATAC)),]
TEST_atac_matrix <- as.data.frame(TEST_ATAC)
shared<-intersect(rownames(TEST_atac_matrix),rownames(Result))
TEST_atac_matrix<-TEST_atac_matrix[shared,]
Result<-Result[shared,]
Result <- as.data.frame(Result)
Result$index <- rownames(Result)
TEST_atac_matrix$index <- rownames(TEST_atac_matrix)
write_feather(TEST_atac_matrix, paste0(save_path,"LIGER_true.feather")) 
write_feather(Result, paste0(save_path,"LIGER_pred.feather")) 