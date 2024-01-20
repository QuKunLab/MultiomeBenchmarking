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

dataset_id = "Dataset1"
save_path = "../results/Horizontal/RNA_ATAC/"  ## path to save results
data_path = "../dataset/Horizontal/RNA_ATAC/"+dataset_id ## path to raw data

for i in range(1,3):
    batch_num = str(i)
    path = data_path+"batch"+batch_num
    cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids']
    X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
    gene_names = pd.read_csv(path+'/RNA/genes.tsv', sep = '\t',  header=None, index_col=None)
    gene_names.columns =  ['gene_ids']
    locals()['adata_RNA'+str(i)] = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
    locals()['adata_RNA'+str(i)].var_names_make_unique()
    del X
    # locals()['adata_RNA'+str(i)].var['modality'] = ['Gene Expression']*eval('adata_RNA'+str(i)).shape[1]
    sc.pp.filter_cells(eval('adata_RNA'+str(i)), min_genes=1)
    sc.pp.filter_genes(eval('adata_RNA'+str(i)), min_cells=1)

    cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids']
    X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
    peak_name = pd.read_csv(path+'/ATAC/peaks.tsv',header=None,index_col=None)
    peak_name.columns = ['peak_ids']
    locals()['adata_ATAC'+str(i)] = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
    del X
    sc.pp.filter_cells(eval('adata_ATAC'+str(i)), min_genes=1)
    sc.pp.filter_genes(eval('adata_ATAC'+str(i)), min_cells=1)

overlap_gene = set(adata_RNA1.var_names).intersection(adata_RNA2.var_names)
adata_RNA1 = adata_RNA1[:,list(overlap_gene)]
adata_RNA2 = adata_RNA2[:,adata_RNA1.var_names]

overlap_peak = set(adata_ATAC1.var_names).intersection(adata_ATAC2.var_names)
adata_ATAC1 = adata_ATAC1[:,list(overlap_peak)]
adata_ATAC2 = adata_ATAC2[:,adata_ATAC1.var_names]

feats_name = {"rna": adata_RNA1.var_names, "atac": adata_ATAC1.var_names}

# READ IN THE COUNT MATRICES
# subsample the original dataset to save the training time
subsample = 10
# scATAC-seq of batch 1
counts_atac1 = adata_ATAC1.X.toarray()
counts_atac1 = scmomat.preprocess(counts_atac1, modality = "ATAC")
del adata_ATAC1
# scATAC-seq of batch 2
counts_atac2 = adata_ATAC2.X.toarray()
counts_atac2 = scmomat.preprocess(counts_atac2, modality = "ATAC")
del adata_ATAC2

# scRNA-seq of batch 1
counts_rna1 = adata_RNA1.X.toarray()
counts_rna1 = scmomat.preprocess(counts_rna1, modality = "RNA", log = False)
del adata_RNA1

# scRNA-seq of batch 2
counts_rna2 = adata_RNA2.X.toarray()
counts_rna2 = scmomat.preprocess(counts_rna2, modality = "RNA", log = False)
del adata_RNA2

# CREATE THE COUNTS OBJECT
counts = {"feats_name": feats_name, "nbatches": 2, "rna":[counts_rna1, counts_rna2], "atac": [counts_atac1, counts_atac2]}
del counts_rna1, counts_rna2
del counts_atac1, counts_atac2
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
# end_time = time.time()
# print("running time: " + str(end_time - start_time) + ' s')
np.savetxt(save_path+dataset_id+"_latent_scMoMaT.csv", np.vstack((zs[0],zs[1])) , delimiter=',')
