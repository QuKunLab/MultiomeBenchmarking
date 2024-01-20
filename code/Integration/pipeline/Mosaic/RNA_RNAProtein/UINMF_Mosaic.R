library(GenomicRanges)
library(rliger)
library(Seurat)
library(stringr)

args <- commandArgs(trailingOnly = TRUE)
# 检查输入文件是否存在
if (!file.exists(args[1])) {
  stop("Input CITE-seq Data does not exist.")
}
if (!file.exists(args[2])) {
  stop("Input RNA data does not exist.")
}

# 打印解析结果
cat("Input file:", args[1], "\n")
cat("Input file:", args[2], "\n")
cat("DataName:", args[3], "\n")


path = args[1]
Cell_name <- read.csv(paste0(path,'/RNA/barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"/RNA/features.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"/RNA/matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = Cell_name$V1
pbmc <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 20, min.features = 1)
shared_rna = pbmc@assays$RNA@counts
rm(pbmc, M)
unshare_adt = read.table(file = paste0(path,"/ADT.csv"),header = 1,sep = ',',row.names = 1)


path = args[2]
Cell_name <- read.csv(paste0(path,'/RNA/barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"/RNA/features.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"/RNA/matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = paste('2_', Cell_name$V1, sep = '')
pbmc <- CreateSeuratObject(counts = M, project = "DOGMA", min.cells = 20, min.features = 1)
rna = pbmc@assays$RNA@counts
rm(pbmc, M)

overlap_gene = intersect(row.names(shared_rna),row.names(rna))
shared_rna = shared_rna[overlap_gene,]
rna = rna[overlap_gene,]

# gc()
# start_time <- Sys.time() # From now on time

# Step2: Selecting the unshared features
liger <- createLiger(list(adt = as.matrix(unshare_adt)))
liger <- normalize(liger)
liger <- selectGenes(liger)
liger@var.genes <- row.names(liger@raw.data$adt)
liger <- scaleNotCenter(liger)
unshared_feats = t(liger@scale.data$adt)
rm(liger)

# #### Step 3: Preprocessing and normalization

# + vscode={"languageId": "r"}
#Create a LIGER object and normalize the shared data.
liger <- createLiger(list(contrl = as.matrix(shared_rna), stim = as.matrix(rna)))
rm(rna,shared_rna)
liger <- normalize(liger)
#Note that when we select the variable genes between the shared features, we use the RNA dataset to select variable shared features.
liger <- selectGenes(liger, var.thresh = 0.1, datasets.use =1 , unshared = FALSE,  unshared.datasets = list(2), unshared.thresh= 0.2)
#Scale the data.
DE_gene = read.csv(paste0(args[3],"/DEgene.csv"),header = FALSE)
liger@var.genes <- DE_gene[[1]]
liger <- scaleNotCenter(liger)
print(dim(liger@scale.data$stim))

# + vscode={"languageId": "r"}
#Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix. 
liger@var.unshared.features[[1]] = paste0('adt_',rownames(unshared_feats))
row.names(unshared_feats) = paste0('adt_',rownames(unshared_feats))
liger@scale.unshared.data[[1]] = unshared_feats
# -

# #### Step 4: Joint Matrix Factorization

# + vscode={"languageId": "r"}
#To factorize the datasets and include the unshared datasets, set the use.unshared parameter to TRUE. 
liger <- optimizeALS(liger, k=30, use.unshared = TRUE, max_iters =30,thresh=1e-10)
# -

# #### Step 5: Quantile Normalization and Joint Clustering

# + vscode={"languageId": "r"}
rna_H = liger@H$rna
atac_H = liger@H$atac

# + vscode={"languageId": "r"}
# Liger suggest this to all downstream 
# norm_H = liger@H.norm

# # + vscode={"languageId": "r"}
# # After factorization, the resulting Liger object can be used in all downstream LIGER functions without adjustment. The default reference dataset for quantile normalization is the larger dataset, but the user should select the higher quality dataset as the reference dataset, even if it is the smaller dataset.
liger <- quantile_norm(liger)
write.table(liger@H.norm, paste0(args[3],"/UINMF.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
# liger <- louvainCluster(liger)
# # -

# # #### Step 6: Visualizations and Downstream processing

# # + vscode={"languageId": "r"}
# liger <- runUMAP(liger)

# # + vscode={"languageId": "r"}
# # Next, we can visualize our returned factorized object by dataset to check the alignment between datasets, as well as by cluster determined in the factorization.

# umap_plots <-plotByDatasetAndCluster(liger, axis.labels = c("UMAP1","UMAP2"), return.plots = TRUE)
# umap_plots[[2]]

# # + vscode={"languageId": "r"}
# #We can quantify the alignment between the two dataset with the `calcAlignment` function.
calcAlignment(liger)
# # -

# # ### Dimensionality reduction result
# write.table(data_comb@H$atac,file = 'atac.csv',sep = ',')
# write.table(data_comb@H$rna,file = 'rna.csv',sep = ',')

# end_time <- Sys.time()
# print(end_time - start_time)