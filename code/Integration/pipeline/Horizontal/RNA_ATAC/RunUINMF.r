library(GenomicRanges)
library(rliger)
library(Seurat)
library(stringr)

dataset_id <- "Dataset1"
data_path <- paste0("../dataset/Horizontal/RNA_ATAC/",dataset_id) ## path to raw data
save_path <- "../results/Horizontal/RNA_ATAC/" ## path to save results

# #### Step1: read data
path = paste0(data_path,'batch1/RNA/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"genes.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = Cell_name$V1
pbmc <- CreateSeuratObject(counts = M, project = dataset_id, min.cells = 3, min.features = 2)
rm(M)
rna1 = pbmc@assays$RNA@counts

path = paste0(data_path,'batch1/ATAC/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"peaks.tsv"),header = F, sep = "\t")
counts.atac.1<-readMM(paste0(path,"matrix.mtx"))
row.names(counts.atac.1) = Gene_name$V1
colnames(counts.atac.1) = Cell_name$V1

path = paste0(data_path,'batch2/RNA/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"genes.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = Cell_name$V1
pbmc <- CreateSeuratObject(counts = M, project = dataset_id, min.cells = 3, min.features = 2)
rm(M)
rna2 = pbmc@assays$RNA@counts

path = paste0(data_path,'batch2/ATAC/')
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"peaks.tsv"),header = F, sep = "\t")
counts.atac.2<-readMM(paste0(path,"matrix.mtx"))
row.names(counts.atac.2) = Gene_name$V1
colnames(counts.atac.2) = Cell_name$V1

inter_peak <- intersect(rownames(counts.atac.1),rownames(counts.atac.2))
counts.atac.2 <- counts.atac.2[inter_peak,]
counts.atac.1 <- counts.atac.1[inter_peak,]

# start_time <- Sys.time() # From now on time
# create the liger object for the unshared bin data
liger_bin <- rliger::createLiger(list(peak1 = as.matrix(counts.atac.1), peak2 = as.matrix(counts.atac.2)), remove.missing = FALSE)
liger_bin <- rliger::normalize(liger_bin)

norm <- cbind(liger_bin@norm.data$peak1,liger_bin@norm.data$peak2)
se = CreateSeuratObject(norm)
vars_2000 <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 2000)
top2000 <- head(VariableFeatures(vars_2000),2000)
top2000_feats <-  norm[top2000,]
liger_bin <- selectGenes(liger_bin)
liger_bin@var.genes <- top2000

#Create a LIGER object and normalize the shared data.
liger_rna <- createLiger(list(rna1 = as.matrix(rna1), rna2 = as.matrix(rna2)))
# rm(rna1, rna2)
liger_rna <- normalize(liger_rna)

#Note that when we select the variable genes between the shared features, we use the RNA dataset to select variable shared features.
liger_rna <- selectGenes(liger_rna, var.thresh = 0.1, datasets.use =1 , unshared = FALSE,  unshared.datasets = list(2), unshared.thresh= 0.2)

liger <- createLiger(list(Batch1 = as.matrix(rbind(rna1,counts.atac.1)), Batch2 = as.matrix((rbind(rna2,counts.atac.2)))))
rm(rna1,counts.atac.1,rna2,counts.atac.2)
liger@norm.data$Batch1 = rbind(liger_rna@norm.data$rna1,liger_bin@norm.data$peak1)
liger@norm.data$Batch2 = rbind(liger_rna@norm.data$rna2,liger_bin@norm.data$peak2)
liger <- selectGenes(liger)
liger@var.genes <- c(liger_rna@var.genes, liger_bin@var.genes)

rm(liger_rna, liger_bin)
#Scale the data.
liger <- scaleNotCenter(liger)

# #### Step 4: Joint Matrix Factorization

#To factorize the datasets and include the unshared datasets, set the use.unshared parameter to TRUE.
liger <- optimizeALS(liger, k=30, use.unshared = FALSE, max_iters =30,thresh=1e-10)

# #### Step 5: Quantile Normalization and Joint Clustering

rna_H = liger@H$Batch1
atac_H = liger@H$Batch2

# Liger suggest this to all downstream
norm_H = liger@H.norm

# After factorization, the resulting Liger object can be used in all downstream LIGER functions without adjustment. The default reference dataset for quantile normalization is the larger dataset, but the user should select the higher quality dataset as the reference dataset, even if it is the smaller dataset.
liger <- quantile_norm(liger)
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

write.table(liger@H.norm,file = paste0(save_path,dataset_id,'_latent_UINMF.csv'),sep = ',')
