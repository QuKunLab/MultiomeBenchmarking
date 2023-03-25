library(rliger)
library(Seurat)
library(plyr)
library(Matrix)
library(Signac)
library(GenomicRanges)
library(SummarizedExperiment)
library(data.table)
library(dplyr)
library(Matrix)
library(BuenColors)
library(stats4)
train_id <- "Dataset35"
test_id <- "Dataset36"
save_path <- "../results/LIGER/"

TRAIN_RNAfile <- paste0("../data/",train_id,"/RNA/")
TRAIN_RNA <- Read10X(data.dir = TRAIN_RNAfile ,gene.column = 1)
TRAIN_ATACfile <- paste0("../data/",train_id,"/ATAC/")
TRAIN_ATAC <- Read10X(data.dir = TRAIN_ATACfile ,gene.column = 1)
TEST_RNAfile <-  paste0("../data/",test_id,"/RNA/")
TEST_RNA <- Read10X(data.dir = TEST_RNAfile ,gene.column = 1)
TEST_ATACfile <- paste0("../data/",test_id,"/ATAC/")
TEST_ATAC <- Read10X(data.dir = TEST_RNA ,gene.column = 1)

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

write.table(Result,paste0(save_path,"LIGER_",train_id,"_",test_id,"_pred.csv"),sep=',',quote=F,col.names = TRUE)
write.table(TEST_atac_matrix,paste0(save_path,"LIGER_",train_id,"_",test_id,"_true.csv"),sep=',',quote=F,col.names = TRUE)

t_TEST_atac_matrix<-t(TEST_atac_matrix)
t_Result<-t(Result)
###by row 
row_corr <- list()
for (i in 1:length(rownames(TEST_atac_matrix))){
    row_corr[[i]]<- cor(t_TEST_atac_matrix[,i],t_Result[,i])
}
row_corr <- as.data.frame(row_corr)
row_corr<- t(row_corr)
rownames(row_corr) <- colnames(t_TEST_atac_matrix)

###by colnames 
col_corr <- list()
for (i in 1:length(colnames(TEST_atac_matrix))){
    col_corr[[i]]<- cor(TEST_atac_matrix[,i],Result[,i])
}
col_corr <- as.data.frame(col_corr)
col_corr<- t(col_corr)
rownames(col_corr) <- colnames(Result)

write.table(row_corr,paste0(save_path,"LIGER_",train_id,"_",test_id,"_peak_pcc.csv"),sep='\t',quote=F,col.names = TRUE)
write.table(col_corr,paste0(save_path,"LIGER_",train_id,"_",test_id,"_cell_pcc.csv"),sep='\t',quote=F,col.names = TRUE)


