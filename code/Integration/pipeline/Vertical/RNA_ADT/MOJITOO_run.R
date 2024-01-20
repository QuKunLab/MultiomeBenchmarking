suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Signac))
suppressPackageStartupMessages(library(ggsci))
library(reticulate)
use_condaenv("/home/math/hyl2016/miniconda3/envs/seuratV4_python38")
suppressPackageStartupMessages(library(MOJITOO))
library(Matrix)
library(magrittr)

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

object <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 0, min.features = 0)
rm(M)
ADT = read.table(file = paste0(args[1],"/ADT.csv"),header = 1,sep = ',',row.names = 1)
colnames(ADT)<-colnames(object)
adt_assay <- CreateAssayObject(counts = ADT)
object[["ADT"]] <- adt_assay
rm(ADT)
rm(adt_assay)

DefaultAssay(object) <- "RNA"
object <- NormalizeData(object)
object <- FindVariableFeatures(object, nfeatures = 4000)
object <- ScaleData(object, verbose=F)
object <- RunPCA(object, npcs=50, reduction.name="RNA_PCA", verbose=F)
DefaultAssay(object) <- "ADT"
VariableFeatures(object) <- rownames(object@assays$ADT@counts)
object <- NormalizeData(object, normalization.method = 'CLR', margin = 2) %>% 
  ScaleData() %>% RunPCA(reduction.name = 'apca', verbose=F)

dimension <- 30
if (dim(object[['apca']])[2]-1 < 30) dimension <- dim(object[['apca']])[2]-1
object <- mojitoo(
     object=object,
     reduction.list = list("RNA_PCA", "apca"),
     dims.list = list(1:30, 1:dimension), ## exclude 1st dimension of LSI
     reduction.name='MOJITOO',
     assay="RNA"
)

DefaultAssay(object) <- "RNA"
embedd <- Embeddings(object[["MOJITOO"]])
object <- RunUMAP(object, reduction="MOJITOO", reduction.name="MOJITOO_UMAP", dims=1:ncol(embedd), verbose=F)

embedd <- Embeddings(object[["MOJITOO"]])
write.table(embedd, file = paste0(args[2], "/MOJITOO.csv"), sep = ",", row.names = FALSE,
            col.names = F)
