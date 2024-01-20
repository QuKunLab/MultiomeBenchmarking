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
  stop("Input ATAC data does not exist.")
}

# 打印解析结果
cat("Input file:", args[1], "\n")
cat("Input file:", args[2], "\n")
cat("DataName:", args[3], "\n")

# # Intergration of RNA+ATAC and ATAC data

# ## read data
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
# ### read scATAC data
path = paste0(args[2],'/ATAC/')
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
DE_gene <- rownames(decomp)[decomp$mean>0.02 & decomp$p.value <= 0.05]        ## These two parameters should be adjusted according to the data set.

# decomp_new = decomp[decomp$p.value <= 0.05,]
# decomp_new <- decomp_new[order(-decomp_new$mean),]
# DE_gene <- rownames(head(decomp_new, 2000))

length(DE_gene)
# sce.rna <- sce.rna[hvgs,]

sce.atac <- SingleCellExperiment(list(counts=shared_atac1))
# Normalise
sce.atac <- logNormCounts(sce.atac)

# Feature selection using highly variable peaks
# And adding matching peaks to genes
decomp <- modelGeneVar(sce.atac)
hvgs <- rownames(decomp)[decomp$mean>0.25
                         & decomp$p.value <= 0.05]          ## These two parameters should be adjusted according to the data set.

# decomp_new = decomp[decomp$p.value <= 0.05,]
# decomp_new <- decomp_new[order(-decomp_new$mean),]
# hvgs <- rownames(head(decomp_new, 10000))

# sce.atac <- sce.atac[hvgs,]

sce.atac2 <- SingleCellExperiment(list(counts=shared_atac2))
# Normalise
sce.atac2 <- logNormCounts(sce.atac2)

# Feature selection using highly variable peaks
# And adding matching peaks to genes
decomp <- modelGeneVar(sce.atac2)
hvgs2 <- rownames(decomp)[decomp$mean>0.25
                         & decomp$p.value <= 0.05]          ## These two parameters should be adjusted according to the data set.

# decomp_new = decomp[decomp$p.value <= 0.05,]
# decomp_new <- decomp_new[order(-decomp_new$mean),]
# hvgs2 <- rownames(head(decomp_new, 10000))

length(hvgs2)
DE_peak = intersect(hvgs, hvgs2)
length(DE_peak)
# sce.atac2 <- sce.atac2[hvgs2,]

# DE_gene = read.csv(paste0(dataname,"/DEgene.csv"),header = FALSE)

logcounts_all = rbind(sce.rna@assays@data@listData$logcounts[DE_gene,], sce.atac@assays@data@listData$logcounts[DE_peak,])
# logcounts_all = rbind(sce.rna@assays@data@listData$logcounts, sce.atac@assays@data@listData$logcounts[DE_peak,])
dim(logcounts_all)
assay_list = list(Multiome = logcounts_all,  ATAC = as(sce.atac2@assays@data@listData$logcounts[DE_peak,],"CsparseMatrix"))

stab = stabMap(assay_list,
               reference_list = c("Multiome"),
               plot = FALSE,maxFeatures=20000)

stab_umap = calculateUMAP(t(stab))
dim(stab_umap)
#Dataset4
plot(stab_umap, pch = 16, cex = 0.3, col = factor(c(rep("Multiome", dim(sce.atac@assays@data@listData$logcounts)[2]), rep("atac", dim(sce.atac2@assays@data@listData$logcounts)[2]))))

write.table(stab, paste0(args[3],"/StapMap.csv"), sep = ",", row.names = FALSE,
            col.names = FALSE)

