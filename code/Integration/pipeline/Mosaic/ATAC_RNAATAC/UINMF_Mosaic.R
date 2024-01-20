library(GenomicRanges)
library(rliger)
library(Seurat)
library(stringr)

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

# ## Scenario 2: Intergration of RNA+ATAC and RNA data

# #### Step1: read data

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

path = paste0(args[2],'/ATAC/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
shared_atac2 <-readMM(paste0(path,"matrix.mtx"))
row.names(shared_atac2) = Gene_name$V1
colnames(shared_atac2) = paste0("batch2_",Cell_name$V1)
gc()


# #### Step2: Selecting the unshared features

liger <- createLiger(list(geneExpression = as.matrix(rna)))
rm(rna)
liger <- normalize(liger)
#Then we scale, but do not center the unshared features
liger <- selectGenes(liger)
DE_gene = read.csv(paste0(args[3],"/DEgene.csv"),header = FALSE)
liger@var.genes <- DE_gene[[1]]
liger <- scaleNotCenter(liger)
unshared_feats = liger@scale.data$geneExpression
rm(liger)
# #### Step 3: Preprocessing and normalization

#Create a LIGER object and normalize the shared data.
liger <- createLiger(list(stim =  as.matrix(shared_atac1), control =  as.matrix(shared_atac2)))
rm(shared_atac2, shared_atac1)
liger <- normalize(liger)

#Note that when we select the variable genes between the shared features, we use the RNA dataset to select variable shared features.
liger <- selectGenes(liger, var.thresh = 0.00001, datasets.use =1 , unshared = FALSE,  unshared.datasets = list(2), unshared.thresh= 0.2)

#Scale the data.
liger <- scaleNotCenter(liger)

#Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix. 
gene_names <- rownames(unshared_feats)
liger@var.unshared.features[[1]] = gene_names
liger@scale.unshared.data[[1]] = t(unshared_feats)
rm(unshared_feats)
# #### Step 4: Joint Matrix Factorization

#To factorize the datasets and include the unshared datasets, set the use.unshared parameter to TRUE. 
liger <- optimizeALS(liger, k=30, use.unshared = TRUE, max_iters =30,thresh=1e-10)

# #### Step 5: Quantile Normalization and Joint Clustering

# Liger suggest this to all downstream 
norm_H = liger@H.norm

# After factorization, the resulting Liger object can be used in all downstream LIGER functions without adjustment. The default reference dataset for quantile normalization is the larger dataset, but the user should select the higher quality dataset as the reference dataset, even if it is the smaller dataset.
liger <- quantile_norm(liger)
write.table(liger@H.norm, paste0(args[3],"/UINMF.csv"), sep = ",", row.names = FALSE, col.names = FALSE)

calcAlignment(liger)

