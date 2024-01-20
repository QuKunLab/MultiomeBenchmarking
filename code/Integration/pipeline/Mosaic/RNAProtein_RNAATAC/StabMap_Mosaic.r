library(StabMap)
library(Seurat)
require(stringr)
require(Seurat)
require(Signac)
require(Matrix)
library(EnsDb.Hsapiens.v86)
set.seed(2021)

library(GenomicRanges)
library(SingleCellExperiment)

library(rliger)
library(Seurat)
library(scran)

args <- commandArgs(trailingOnly = TRUE)
# 检查输入文件是否存在
if (!file.exists(args[1])) {
  stop("Input 10xMultiome Data does not exist.")
}
if (!file.exists(args[2])) {
  stop("Input CITE-seq data does not exist.")
}

# 打印解析结果
cat("Input file:", args[1], "\n")
cat("Input file:", args[2], "\n")
cat("DataName:", args[3], "\n")
# #Intergration of RNA+ATAC and RNA+Protein data

# ##read data
# ### read RNA+ATAC data
path = paste0(args[1],'/RNA/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = paste0("batch1_",Cell_name$V1)
pbmc <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 20, min.features = 1)
rm(M)

path = paste0(args[1],'/ATAC/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
atac_counts<-readMM(paste0(path,"matrix.mtx"))
row.names(atac_counts) = Gene_name$V1
colnames(atac_counts) = paste0("batch1_",Cell_name$V1)
chrom_assay <- CreateChromatinAssay(
   counts = atac_counts,
   sep = c("-", "-"),
   genome = NULL,
   fragments = NULL,
   min.cells = 1,
   annotation = NULL
 )
pbmc[["ATAC"]] <- chrom_assay
rm(atac_counts, chrom_assay)

# ### read RNA+Protein data
path = paste0(args[2],'/RNA/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = Cell_name$V1
pbmc2 <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 20, min.features = 1)
rm(M)

ADT = read.table(file = paste0(path0,'/ADT.csv'),header = 1,sep = ',',row.names = 1)
ADT <- CreateSeuratObject(counts = ADT, project = "DOGMA", min.cells = 1, min.features = 1)

sce.rna <- SingleCellExperiment(list(counts=pbmc@assays$RNA@counts))
# Normalisation
sce.rna <- logNormCounts(sce.rna)
# Feature selection
decomp_1 <- modelGeneVar(sce.rna)
DE_gene1 <- rownames(decomp_1)[decomp_1$mean>0.001 & decomp_1$p.value <= 0.4]         ## These two parameters should be adjusted according to the data set.

sce.atac <- SingleCellExperiment(list(counts=pbmc@assays$ATAC@counts))
# Normalise
sce.atac <- logNormCounts(sce.atac)

# Feature selection using highly variable peaks
# And adding matching peaks to genes
decomp <- modelGeneVar(sce.atac)
DE_peak <- rownames(decomp)[decomp$mean>0.1
                         & decomp$p.value <= 0.05]          ## These two parameters should be adjusted according to the data set.

sce.rna2 <- SingleCellExperiment(list(counts=pbmc2@assays$RNA@counts))
# Normalisation
sce.rna2 <- logNormCounts(sce.rna2)
# Feature selection
decomp2 <- modelGeneVar(sce.rna2)
DE_gene2 <- rownames(decomp2)[decomp2$mean>0.001 & decomp2$p.value <= 0.4]          ## These two parameters should be adjusted according to the data set.

length(DE_gene1)
length(DE_gene2)
DE_gene = intersect(DE_gene1, DE_gene2)
length(DE_gene)

sce.adt <- SingleCellExperiment(list(counts=ADT@assays$RNA@counts))
# Normalisation
sce.adt <- logNormCounts(sce.adt)

# DE_gene = read.csv(paste0(dataname,"/DEgene.csv"),header = FALSE)

logcounts_all = rbind(sce.rna@assays@data@listData$logcounts[DE_gene,], 
                      sce.atac@assays@data@listData$logcounts[DE_peak,])
CITE = rbind(as(sce.rna2@assays@data@listData$logcounts[DE_gene,],"CsparseMatrix"), 
                      sce.adt@assays@data@listData$logcounts)

assay_list = list(Multiome = logcounts_all,
                  CITE = CITE)

# rm(pbmc, pbmc2, ADT)
lapply(assay_list, dim)
lapply(assay_list, class)

stab = stabMap(assay_list,
               reference_list = c("Multiome"),
               plot = FALSE,maxFeatures=20000)

stab_umap = calculateUMAP(t(stab))
dim(stab_umap)
#Dataset4
plot(stab_umap, pch = 16, cex = 0.3, 
     col = factor(c(rep("Multiome", dim(sce.rna@assays@data@listData$logcounts)[2]),
     rep("atac", dim(sce.rna2@assays@data@listData$logcounts)[2]))))

DE_gene1 <- rownames(decomp_1)[decomp_1$mean>0.001 & decomp_1$p.value <= 0.4]
DE_gene2 <- rownames(decomp2)[decomp2$mean>0.001 & decomp2$p.value <= 0.4]

length(DE_gene1)
length(DE_gene2)
DE_gene = intersect(DE_gene1, DE_gene2)
length(DE_gene)

write.table(stab, paste0(args[3],"/StapMap.csv"), sep = ",", row.names = FALSE,
            col.names = FALSE)