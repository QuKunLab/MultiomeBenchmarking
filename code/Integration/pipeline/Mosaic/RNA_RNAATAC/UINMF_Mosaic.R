library(GenomicRanges)
library(rliger)
library(Seurat)
library(stringr)

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

# ## Scenario 2: Intergration of RNA+ATAC and RNA data

# #### Step1: read data

path = paste0(args[1],'/RNA/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = Cell_name$V1
pbmc <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 20, min.features = 1)
rm(M)
rna = pbmc@assays$RNA@counts

path = paste0(args[1],'/ATAC/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
unshared_atac<-readMM(paste0(path,"matrix.mtx"))
row.names(unshared_atac) = Gene_name$V1
colnames(unshared_atac) = Cell_name$V1

path = paste0(args[2],'/RNA/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = paste0("batch2_",Cell_name$V1)
pbmc <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 20, min.features = 1)
rm(M)
shared_atac = pbmc@assays$RNA@counts

se = CreateSeuratObject(unshared_atac)
vars_2000 <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 2000)
top2000 <- head(VariableFeatures(vars_2000),2000)
DE_peak = VariableFeatures(vars_2000)


DE_gene = read.csv(paste0(args[3],"/DEgene.csv"),header = FALSE)
#DE_peak = read.csv(paste0(args[3],"/DEpeak.csv"),header = FALSE)

# #### Step2: Selecting the unshared features

liger <- createLiger(list(peaks = as.matrix(unshared_atac)))
rm(unshared_atac)
liger <- normalize(liger)
#Then we scale, but do not center the unshared features
#liger <- selectGenes(liger)
liger <- selectGenes(liger, var.thresh = 0.00001, datasets.use =1 , unshared = FALSE)
liger@var.genes <- DE_peak
liger <- scaleNotCenter(liger)
unshared_feats = liger@scale.data$peaks
rm(liger)

# #### Step 3: Preprocessing and normalization

#Create a LIGER object and normalize the shared data.
liger <- createLiger(list(rna = as.matrix(rna), atac = as.matrix(shared_atac)))
rm(shared_atac, rna)
liger <- normalize(liger)

#Note that when we select the variable genes between the shared features, we use the RNA dataset to select variable shared features.
liger <- selectGenes(liger, var.thresh = 0.1, datasets.use =1 , unshared = FALSE,  unshared.datasets = list(2), unshared.thresh= 0.2)

# liger@var.genes <- DE_gene[[1]]
#Scale the data.
liger <- scaleNotCenter(liger)

#Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix. 
peak_names <- rownames(unshared_feats)
liger@var.unshared.features[[1]] = peak_names
liger@scale.unshared.data[[1]] = t(unshared_feats)
rm(unshared_feats)

# #### Step 4: Joint Matrix Factorization

#To factorize the datasets and include the unshared datasets, set the use.unshared parameter to TRUE. 
liger <- optimizeALS(liger, k=30, use.unshared = TRUE, max_iters =30,thresh=1e-10)

# #### Step 5: Quantile Normalization and Joint Clustering

rna_H = liger@H$rna
atac_H = liger@H$atac

# Liger suggest this to all downstream 

# After factorization, the resulting Liger object can be used in all downstream LIGER functions without adjustment. The default reference dataset for quantile normalization is the larger dataset, but the user should select the higher quality dataset as the reference dataset, even if it is the smaller dataset.
liger <- quantile_norm(liger)

write.table(liger@H.norm, paste0(args[3],"/UINMF.csv"), sep = ",", row.names = FALSE,
            col.names = FALSE)

# liger <- louvainCluster(liger)

# #### Step 6: Visualizations and Downstream processing

# liger <- runUMAP(liger)

# +
# Next, we can visualize our returned factorized object by dataset to check the alignment between datasets, as well as by cluster determined in the factorization.

# umap_plots <-plotByDatasetAndCluster(liger, axis.labels = c("UMAP1","UMAP2"), return.plots = TRUE)
# umap_plots[[2]]
# -

#We can quantify the alignment between the two dataset with the `calcAlignment` function.
calcAlignment(liger)

# ### Dimensionality reduction result

# write.table(data_comb@H$atac,file = 'atac.csv',sep = ',')
# write.table(data_comb@H$rna,file = 'rna.csv',sep = ',')


