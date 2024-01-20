suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Signac))
suppressPackageStartupMessages(library(ggsci))
library(reticulate)
#use_condaenv("/home/hyl2016/.conda/envs/R4")
suppressPackageStartupMessages(library(MOJITOO))
library(Matrix)

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
object <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 3, min.features = 2)

path = paste0(args[1],'/ATAC/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
atac_counts<-readMM(paste0(path,"matrix.mtx"))
row.names(atac_counts) = Gene_name$V1
colnames(atac_counts) = Cell_name$V1
DE_peak = read.csv(paste0(args[2],"/DEpeak.csv"),header = TRUE)
atac_counts <- atac_counts[DE_peak[,2],]

atac_counts <- atac_counts[which(startsWith(rownames(atac_counts), "chr")), ]

chrom_assay <- CreateChromatinAssay(
   counts = atac_counts,
   sep = c("-", "-"),
   genome = NULL,
   fragments = NULL,
   min.cells = 10
)
object[["peak"]] <- chrom_assay
rm(chrom_assay)

#meta <- read.csv(paste0(args[1],'/metadata.csv'), sep=",")
#meta = meta[meta$stim=='Control','predicted.celltype.l2']
#object$celltype <- meta

DefaultAssay(object) <- "RNA"
object <- NormalizeData(object)
object <- FindVariableFeatures(object, nfeatures = 4000)
object <- ScaleData(object, verbose=F)
object <- RunPCA(object, npcs=50, reduction.name="RNA_PCA", verbose=F)
## ATAC pre-processing and LSI dimension reduction
DefaultAssay(object) <- "peak"
object <- RunTFIDF(object, verbose=F)
object <- FindTopFeatures(object, min.cutoff = 'q0')
object <- RunSVD(object, verbose=F)

dimension <- 50
if (dim(object[['lsi']])[2]-1 < 50) dimension <- dim(object[['lsi']])[2]-1

start_time <- Sys.time()

object <- mojitoo(
     object=object,
     reduction.list = list("RNA_PCA", "lsi"),
     dims.list = list(1:50, 2:dimension), ## exclude 1st dimension of LSI
     reduction.name='MOJITOO',
     assay="RNA"
)
DefaultAssay(object) <- "RNA"
embedd <- Embeddings(object[["MOJITOO"]])
write.table(embedd, file = paste0(args[2], "MOJITOO.csv"), sep = ",", row.names = FALSE,col.names = F)
end_time <- Sys.time()
print(end_time - start_time)

