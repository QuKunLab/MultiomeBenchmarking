library(CiteFuse)
library(scater)
library(SingleCellExperiment)
library(DT)
library(Matrix)
library(Seurat)

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
control_rna = pbmc@assays$RNA@counts
rm(M)

control_adt = read.table(file = paste0(args[1],"/ADT.csv"),header = 1,sep = ',',row.names = 1)
control_adt = as(as.matrix(control_adt),"dgCMatrix")
colnames(control_adt) = colnames(control_rna)

CITEseq_example = list(RNA=control_rna, ADT=control_adt)
rm(control_rna)
rm(control_adt)

sce_citeseq <- preprocessing(CITEseq_example)
sce_citeseq <- scater::logNormCounts(sce_citeseq)
sce_citeseq <- normaliseExprs(sce_citeseq, altExp_name = "ADT", transform = "log")
sce_citeseq <- CiteFuse(sce_citeseq)

.normalize <- function(X) {
    row.sum.mdiag <- rowSums(X) - diag(X)
    row.sum.mdiag[row.sum.mdiag == 0] <- 1
    X <- X/(2 * (row.sum.mdiag))
    diag(X) <- 0.5
    return(X)
}

normalized.mat <- metadata(sce_citeseq)[["SNF_W"]]
diag(normalized.mat) <- stats::median(as.vector(normalized.mat))
normalized.mat <- .normalize(normalized.mat)
normalized.mat <- normalized.mat + t(normalized.mat)

binary.mat <- dbscan::sNN(stats::as.dist(0.5 - normalized.mat),
                      k = 20)
distance <- vapply(seq_len(nrow(normalized.mat)),
                     function(x) {
                         tmp <- rep(0, ncol(normalized.mat))
                         tmp[binary.mat$id[x,]] <- 1
                         tmp
                     }, numeric(ncol(normalized.mat)))
distance <- distance + t(distance)
distance[distance>0] = 1
connect <- vapply(seq_len(nrow(normalized.mat)),
                     function(x) {
                         tmp <- rep(0, ncol(normalized.mat))
                         tmp[binary.mat$id[x,]] <- binary.mat$dist[x,]
                         tmp
                     }, numeric(ncol(normalized.mat)))
writeMM(as(connect,'sparseMatrix'),paste0(args[2], "/CiteFuse_connectivities.mtx"))
writeMM(as(distance,'sparseMatrix'),paste0(args[2], "/CiteFuse_distance.mtx"))
