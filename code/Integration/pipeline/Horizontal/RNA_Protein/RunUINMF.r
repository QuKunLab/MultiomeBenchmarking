library(GenomicRanges)
library(rliger)
library(Seurat)
library(stringr)

dataset_id <- "Dataset7"

data_path <- paste0("../dataset/Horizontal/RNA_Protein/",dataset_id) ## path to raw data
save_path <- "../results/Horizontal/RNA_Protein/" ## path to save results

# read data
path <- paste0(data_path,"/batch1/RNA/")
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"genes.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = Cell_name$V1
Dataset <- CreateSeuratObject(counts = M, project = "Dataset", min.cells = 3, min.features = 2)
rna_1 = Dataset@assays$RNA@counts

path <- paste0(data_path,"/batch2/RNA/")
Cell_name <- read.csv(paste0(path,'barcodes.tsv'),header = F)
Gene_name <- read.table(paste0(path,"genes.tsv"),header = F, sep = "\t")
M<-readMM(paste0(path,"matrix.mtx"))
row.names(M) = Gene_name$V1
colnames(M) = Cell_name$V1
Dataset <- CreateSeuratObject(counts = M, project = "Dataset", min.cells = 3, min.features = 2)
rna_2 = Dataset@assays$RNA@counts

adt_1 = read.table(file = paste0(data_path,"/batch1/ADT/ADT.csv"),header = 1,sep = ',',row.names = 1)
adt_2 = read.table(file = paste0(data_path,"/batch2/ADT/ADT.csv"),header = 1,sep = ',',row.names = 1)

liger <- createLiger(list(adt = as.matrix(adt_1)))
liger <- normalize(liger)
liger <- selectGenes(liger)
# liger@var.genes <- row.names(adt_1)
liger@var.genes <- row.names(liger@raw.data$adt)
liger <- scaleNotCenter(liger)
adt_1 = t(liger@scale.data$adt)

liger <- createLiger(list(adt = as.matrix(adt_2)))
liger <- normalize(liger)
liger <- selectGenes(liger)
# liger@var.genes <- row.names(adt_2)
liger@var.genes <- row.names(liger@raw.data$adt)
liger <- scaleNotCenter(liger)
adt_2 = t(liger@scale.data$adt)

liger <- createLiger(list(batch1 = rna_1, batch2 = rna_2))
liger <- normalize(liger)
#Note that when we select the variable genes between the shared features, we use the RNA dataset to select variable shared features.
liger <- selectGenes(liger, var.thresh = 0.1, datasets.use =1 , unshared = FALSE,  unshared.datasets = list(2), unshared.thresh= 0.2)
#Scale the data.
liger <- scaleNotCenter(liger)

#Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix.
liger@var.unshared.features[[1]] = paste0('adt_',row.names(adt_1))
row.names(adt_1) = paste0('adt_',row.names(adt_1))
liger@scale.unshared.data[[1]] = adt_1

liger@var.unshared.features[[2]] = paste0('adt_',row.names(adt_2))
row.names(adt_2) = paste0('adt_',row.names(adt_2))
liger@scale.unshared.data[[2]] = adt_2

#To factorize the datasets and include the unshared datasets, set the use.unshared parameter to TRUE.
liger <- optimizeALS(liger, k=30, use.unshared = TRUE, max_iters =30,thresh=1e-10)

rna_H = liger@H$rna
liger <- quantile_norm(liger)
calcAlignment(liger)

write.table(liger@H.norm,file = paste0(save_path,dataset_id,'_latent_UINMF.csv'),sep = ",")
