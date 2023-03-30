library(Seurat)
library(ggplot2)
library(Matrix)
library(Signac)
library(stats4)
library(dplyr)
library(dplyr)
library(Matrix)

#####loading data 

train_id <- "Dataset35"
test_id <- "Dataset36"

save_path <- "../results/Seurat/"

TRAIN_RNAfile <- paste0("../data/",train_id,"/RNA/")
TRAIN_RNA <- Read10X(data.dir = TRAIN_RNAfile ,gene.column = 1)

TRAIN_ATACfile <- paste0("../data/",train_id,"/ATAC/")
TRAIN_ATAC <- Read10X(data.dir = TRAIN_ATACfile ,gene.column = 1)

TEST_RNAfile <-  paste0("../data/",test_id,"/RNA/")
TEST_RNA <- Read10X(data.dir = TEST_RNAfile ,gene.column = 1)

TEST_ATACfile <- paste0("../data/",test_id,"/ATAC/")
TEST_ATAC <- Read10X(data.dir = TEST_RNA ,gene.column = 1)

TEST_RNA_obj <- CreateSeuratObject(counts = TEST_RNA, assay="RNA",project = test_id, min.cells = 1, min.features = 10)
TEST_RNA_obj <- NormalizeData(TEST_RNA_obj, normalization.method = "LogNormalize", scale.factor = 1e4)

TRAIN_RNA_obj <- CreateSeuratObject(counts = TRAIN_RNA, assay="RNA",project = train_id, min.cells = 1, min.features = 10)
TRAIN_RNA_obj <- NormalizeData(TRAIN_RNA_obj, normalization.method = "LogNormalize", scale.factor = 1e4) 
TRAIN_RNA_obj <- FindVariableFeatures(TRAIN_RNA_obj, selection.method = 'vst', nfeatures = 4000)

ATAC_chromassay <- CreateChromatinAssay(counts = TRAIN_ATAC,sep = c("-", "-"),min.cells = 5,min.features = 10)
TRAIN_ATAC_obj <- CreateSeuratObject(counts = ATAC_chromassay, assay="ATAC",project = train_id, min.cells = 5, min.features = 10)

ATAC_chromassay <- CreateChromatinAssay(counts = TEST_ATAC,sep = c("-", "-"),min.cells = 5,min.features = 10)
TEST_ATAC_obj <- CreateSeuratObject(counts = ATAC_chromassay, assay="ATAC",project = test_id , min.cells = 5, min.features = 10)

features<-TRAIN_RNA_obj@assays$RNA@var.features

#####training

DN = 30

options (warn = -1)

anchors <- FindTransferAnchors(reference =TRAIN_RNA_obj,query = TEST_RNA_obj,reduction = 'cca',features=features,reference.assay = 'RNA',query.assay = 'RNA', k.filter = NA, dims = 1:DN)
##atac
refdata <- GetAssayData(object = TRAIN_ATAC_obj,assay = 'ATAC',slot = 'data')

imputation <- TransferData(anchorset = anchors,refdata = refdata,weight.reduction = 'cca',dims = 1:DN,k.weight=10)

options(warn = -1)

Imp_New_genes = as.data.frame(imputation@data)

test_mod <- as.data.frame(TEST_ATAC_obj@assays$ATAC@counts)#counts

shared<-intersect(rownames(test_mod),rownames(Imp_New_genes))

Imp_New_genes<-Imp_New_genes[shared,]

test_mod<-test_mod[shared,]

write.table(Imp_New_genes,paste0(save_path,"Seurat_",train_id,"_",test_id,"_pred.csv"),sep=',',quote=F,col.names = TRUE)
write.table(test_mod,paste0(save_path,"Seurat_",train_id,"_",test_id,"_true.csv"),sep=',',quote=F,col.names = TRUE)

t_test_mod<-t(test_mod)
t_Imp_New_genes<-t(Imp_New_genes)
row_corr <- list()
for (i in 1:length(rownames(test_mod))){
    row_corr[[i]]<- cor(t_test_mod[,i],t_Imp_New_genes[,i])
}
row_corr <- as.data.frame(row_corr)
row_corr<- t(row_corr)
rownames(row_corr) <- colnames(t_test_mod)

col_corr <- list()
for (i in 1:length(colnames(test_mod))){
    col_corr[[i]]<- cor(test_mod[,i],Imp_New_genes[,i])
}
col_corr <- as.data.frame(col_corr)
col_corr<- t(col_corr)
rownames(col_corr) <- colnames(test_mod)

write.table(row_corr,paste0(save_path,"Seurat_",train_id,"_",test_id,"_peak_pcc.csv"),sep='\t',quote=F,col.names = TRUE)
write.table(col_corr,paste0(save_path,"Seurat_",train_id,"_",test_id,"_cell_pcc.csv"),sep='\t',quote=F,col.names = TRUE)
