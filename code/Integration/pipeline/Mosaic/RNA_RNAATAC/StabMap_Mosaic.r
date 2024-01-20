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

if (!file.exists(args[1])) {
  stop("Input 10xMultiome Data does not exist.")
}
if (!file.exists(args[2])) {
  stop("Input RNA data does not exist.")
}

# 打印解析结果
cat("Input file:", args[1], "\n")
cat("Input file:", args[2], "\n")
cat("DataName:", args[3], "\n")

# # Scenario 2: Intergration of RNA+ATAC and RNA data

# ## Step1: read data
# ### read RNA+ATAC data
path = paste0(args[1],'/RNA/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = Cell_name$V1
pbmc <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 3, min.features = 2)
rm(M)
rna = pbmc@assays$RNA@counts
rm(pbmc)

path = paste0(args[1],'/ATAC/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = Cell_name$V1
pbmc <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 3, min.features = 2)
rm(M)
shared_atac1 = pbmc@assays$RNA@counts
rm(pbmc)

# ### read scRNA data
path = paste0(args[2],'/RNA/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
shared_atac2 <-readMM(paste0(path,"matrix.mtx"))
row.names(shared_atac2) = Gene_name$V1
colnames(shared_atac2) = paste0("batch2_",Cell_name$V1)
gc()

sce.rna <- SingleCellExperiment(list(counts=rna))
# Normalisation
sce.rna <- logNormCounts(sce.rna)
# Feature selection
decomp <- modelGeneVar(sce.rna)
DE_gene1 <- rownames(decomp)[decomp$mean>0.01 & decomp$p.value <= 0.05]          ## These two parameters should be adjusted according to the data set.

sce.atac <- SingleCellExperiment(list(counts=shared_atac1))
# Normalise
sce.atac <- logNormCounts(sce.atac)

# Feature selection using highly variable peaks
# And adding matching peaks to genes
decomp <- modelGeneVar(sce.atac)
DE_peak <- rownames(decomp)[decomp$mean>0.01
                         & decomp$p.value <= 0.05]                     ## These two parameters should be adjusted according to the data set.

sce.rna2 <- SingleCellExperiment(list(counts=shared_atac2))
# Normalisation
sce.rna2 <- logNormCounts(sce.rna2)
# Feature selection
decomp2 <- modelGeneVar(sce.rna2)
DE_gene2 <- rownames(decomp2)[decomp2$mean>0.01 & decomp2$p.value <= 0.05]          ## These two parameters should be adjusted according to the data set.

length(DE_gene1)
length(DE_gene2)
DE_gene = intersect(DE_gene1, DE_gene2)
length(DE_gene)


# DE_gene = read.csv(paste0(dataname,"/DEgene.csv"),header = FALSE)

logcounts_all = rbind(sce.rna@assays@data@listData$logcounts[DE_gene,], 
                      sce.atac@assays@data@listData$logcounts[DE_peak,])
# logcounts_all = rbind(sce.rna@assays@data@listData$logcounts, sce.atac@assays@data@listData$logcounts[DE_peak,])
dim(logcounts_all)
assay_list = list(Multiome = logcounts_all,
                  ATAC = as(sce.rna2@assays@data@listData$logcounts[DE_gene,],"CsparseMatrix"))

stab = stabMap(assay_list,
               reference_list = c("Multiome"),
               plot = FALSE,maxFeatures=20000)

stab_umap = calculateUMAP(t(stab))
dim(stab_umap)
#Dataset4
plot(stab_umap, pch = 16, cex = 0.3, 
     col = factor(c(rep("Multiome", dim(sce.rna@assays@data@listData$logcounts)[2]),
     rep("atac", dim(sce.rna2@assays@data@listData$logcounts)[2]))))

write.table(stab, paste0(args[3],"/StapMap.csv"), sep = ",", row.names = FALSE,
            col.names = FALSE)
