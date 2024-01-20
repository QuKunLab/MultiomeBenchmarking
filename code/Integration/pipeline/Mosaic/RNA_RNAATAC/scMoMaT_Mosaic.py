from scipy.io import mmread
from scipy.sparse import csr_matrix
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import time
import torch
import scmomat 
import sys, os

arguments = sys.argv

if not os.path.exists(arguments[2]):
    print("Input 10xMultiome Data does not exist.")

if not os.path.exists(arguments[1]):
    print("Input RNA Data does not exist.")
    
if not os.path.exists(arguments[3]):
    print("Don't input DataName.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")
print("Input file: "+ arguments[2] + " \n")
print("Data Name: "+ arguments[3] + " \n")

## 10x multiome
path = arguments[1]
# gene expression
cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_names = pd.read_csv(path+'/RNA/features.tsv', sep = '\t',  header=None, index_col=None) 
gene_names.columns =  ['gene_ids'] 
adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA.var_names_make_unique()
del X
sc.pp.filter_cells(adata_RNA, min_genes=1)
sc.pp.filter_genes(adata_RNA, min_cells=1)

# peak information
cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
peak_name = pd.read_csv(path+'/ATAC/features.tsv',header=None,index_col=None)
peak_name.columns = ['peak_ids']
adata_ATAC  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
del X
sc.pp.filter_cells(adata_ATAC, min_genes=1)
sc.pp.filter_genes(adata_ATAC, min_cells=1)

path = arguments[2]
## scRNA-seq
cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
peak_name = pd.read_csv(path+'/RNA/features.tsv',header=None,index_col=None)
peak_name.columns = ['peak_ids']
adata_rna  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
del X
adata_rna.var['modality'] = ['Gene Expression']*adata_rna.shape[1]
sc.pp.filter_cells(adata_rna, min_genes=1)
sc.pp.filter_genes(adata_rna, min_cells=1)

overlap_gene = set(adata_rna.var_names).intersection(adata_RNA.var_names)
adata_RNA = adata_RNA[:,list(overlap_gene)]
adata_rna = adata_rna[:,adata_RNA.var_names]

# Df = pd.read_csv(arguments[3]+"/DEgene.csv",index_col=None,header=None)
# idx = [ing in set(Df.values.flatten())  for ing in adata_RNA.var_names]
# adata_RNA = adata_RNA[:,idx]
# adata_rna = adata_rna[:,idx]

#Df = pd.read_csv(arguments[3]+"/DEpeak.csv",index_col=None,header=None)
#idx = [ing in set(Df.values.flatten())  for ing in adata_ATAC.var_names]
#adata_ATAC=adata_ATAC[:,idx]

# obtain the feature name
# regions = adata_ATAC.var_names.values.squeeze()
feats_name = {"rna": adata_RNA.var_names, "atac": adata_ATAC.var_names}

# READ IN THE COUNT MATRICES
# subsample the original dataset to save the training time
subsample = 10
# scATAC-seq of batch 1
counts_atac1 = adata_ATAC.X.toarray()
counts_atac1 = scmomat.preprocess(counts_atac1, modality = "ATAC")
del adata_ATAC
# scATAC-seq of batch 3

# scRNA-seq of batch 1
counts_rna1 = adata_RNA.X.toarray()
counts_rna1 = scmomat.preprocess(counts_rna1, modality = "RNA", log = False)
del adata_RNA

# scRNA-seq of batch 2
counts_rna2 = adata_rna.X.toarray()
counts_rna2 = scmomat.preprocess(counts_rna2, modality = "RNA", log = False)
del adata_rna

counts_rnas = [counts_rna1, counts_rna2]

# CREATE THE COUNTS OBJECT
counts = {"feats_name": feats_name, "nbatches": 2, "rna":counts_rnas, "atac": [counts_atac1, None]}

## Step 2: training scMoMaT
#------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Number of latent dimensions, key hyper-parameter, 20~30 works for most of the cases.
K = 30
#------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Here we list other parameters in the function for illustration purpose, most of these parameters are set as default value.
# weight on regularization term, default value
lamb = 0.001 
# number of total iterations, default value
T = 4000
# print the result after each ``interval'' iterations, default value
interval = 1000
# batch size for each iteraction, default value
batch_size = 0.1
# learning rate, default value
lr = 1e-2
# random seed, default value
seed = 0
# running device, can be CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#------------------------------------------------------------------------------------------------------------------------------------


model = scmomat.scmomat_model(counts = counts, K = K, batch_size = batch_size, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
losses = model.train_func(T = T)
zs = model.extract_cell_factors() #this is a list
np.savetxt(arguments[3]+"/scMoMaT.csv", np.vstack((zs[0],zs[1])) , delimiter=',')
