library(GenomicRanges)
library(rliger)
library(Seurat)
library(stringr)

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
# ## Scenario 1: Intergration of RNA+ATAC and RNA+Protein data

# #### Step1: read data

path = paste0(args[1],"/RNA/")
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = paste0("batch1_",Cell_name$V1)
pbmc <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 1, min.features = 1)
rna = pbmc@assays$RNA@counts
rm(M, pbmc)

path = paste0(args[1],"/ATAC/")
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
unshared_atac<-readMM(paste0(path,"matrix.mtx"))
row.names(unshared_atac) = Gene_name$V1
colnames(unshared_atac) = paste0("batch1_",Cell_name$V1)
pbmc <- CreateSeuratObject(counts = unshared_atac, project = "DOGMA", min.cells = 1, min.features = 1)
unshared_atac = pbmc@assays$RNA@counts
rm(pbmc)

path = paste0(args[2],"/RNA/")
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"features.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = Cell_name$V1
pbmc <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 1, min.features = 2)
shared_rna = pbmc@assays$RNA@counts
rm(M, pbmc)

unshared_adt = read.table(file = paste0(args[2],"/ADT.csv"),header = 1,sep = ',',row.names = 1)

gc()
# start_time <- Sys.time() # From now on time
# #### Step2: Selecting the unshared features
DE_gene = read.csv(paste0(args[3],"/DEgene.csv"),header = FALSE)
# DE_peak = read.csv(paste0(args[3],"/DEpeak.csv"),header = FALSE)

liger <- createLiger(list(peaks = unshared_atac))
liger <- normalize(liger)
# norm <- liger@norm.data$peaks
# +
#Then we select the top 2,000 variable features:
# se = CreateSeuratObject(norm)
# # vars_2000 <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 30000)
# # top2000 <- head(VariableFeatures(vars_2000),30000)
# top2000_feats <-  norm[DE_peak[[1]],]  
# rm(se)

#Then we scale, but do not center the unshared features
liger <- selectGenes(liger, var.thresh = 0.00001, datasets.use =1 , unshared = FALSE)
# liger <- selectGenes(liger)
# liger@var.genes <- DE_peak[[1]]
liger <- scaleNotCenter(liger)
unshared_atac = t(liger@scale.data$peaks)
# -

liger <- createLiger(list(adt = as.matrix(unshared_adt)))
liger <- normalize(liger)
liger <- selectGenes(liger)
liger@var.genes <- row.names(liger@raw.data$adt)
liger <- scaleNotCenter(liger)
unshared_adt = t(liger@scale.data$adt)

# #### Step 3: Preprocessing and normalization


#Create a LIGER object and normalize the shared data.
liger <- createLiger(list(stim = as.matrix(rna), contrl = as.matrix(shared_rna)))
rm(rna,shared_rna)
liger <- normalize(liger)

#Note that when we select the variable genes between the shared features, we use the RNA dataset to select variable shared features.
liger <- selectGenes(liger, var.thresh = 0.1, datasets.use =1 , unshared = FALSE,  unshared.datasets = list(2), unshared.thresh= 0.2)
# liger@var.genes <- DE_gene[[1]]
#Scale the data.
liger <- scaleNotCenter(liger)

#Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix. 
se = CreateSeuratObject(unshared_atac)
vars_2000 <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 2000)
top2000 <- head(VariableFeatures(vars_2000),2000)

liger@var.unshared.features[[1]] = rownames(unshared_atac[VariableFeatures(vars_2000),])
liger@scale.unshared.data[[1]] = unshared_atac[VariableFeatures(vars_2000),]

#Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix. 
liger@var.unshared.features[[2]] = paste0('adt_',row.names(unshared_adt))
row.names(unshared_adt) = paste0('adt_',row.names(unshared_adt))
liger@scale.unshared.data[[2]] = unshared_adt

# #### Step 4: Joint Matrix Factorization

#To factorize the datasets and include the unshared datasets, set the use.unshared parameter to TRUE. 
liger <- optimizeALS(liger, k=30, use.unshared = TRUE, max_iters =30,thresh=1e-10)

# #### Step 5: Quantile Normalization and Joint Clustering

rna_H = liger@H$rna
atac_H = liger@H$atac

# Liger suggest this to all downstream 

# print(norm_H)


# After factorization, the resulting Liger object can be used in all downstream LIGER functions without adjustment. The default reference dataset for quantile normalization is the larger dataset, but the user should select the higher quality dataset as the reference dataset, even if it is the smaller dataset.
liger <- quantile_norm(liger)
write.table(liger@H.norm, paste0(args[3],"/UINMF.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
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

# end_time <- Sys.time()
# print(end_time - start_time)
