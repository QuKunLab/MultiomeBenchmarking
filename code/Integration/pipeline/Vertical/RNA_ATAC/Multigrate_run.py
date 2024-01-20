import scarches as sca
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
import muon
import gdown
import json
import time
import os
import sys

import warnings
warnings.filterwarnings("ignore")

arguments = sys.argv

if not os.path.exists(arguments[1]):
    print("Input Data does not exist.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")

data_path = arguments[1]

RNA_path = data_path+"/RNA/"
cell_names = pd.read_csv(RNA_path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
cell_names['cell_ids'] = cell_names['cell_ids'].str.replace('.','-')
X = csr_matrix(mmread(RNA_path+'matrix.mtx').T)
gene_names = pd.read_csv(RNA_path+'features.tsv', sep = '\t',  header=None, index_col=None) 
gene_names.columns =  ['gene_ids'] 
adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA.var_names_make_unique()
del X

adata_RNA.layers["counts"] = adata_RNA.X.copy()
sc.pp.normalize_total(adata_RNA)
sc.pp.log1p(adata_RNA)
sc.pp.highly_variable_genes(
    adata_RNA,
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=False
)
adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable].copy()

ATAC_path = data_path+"/ATAC/"
cell_names = pd.read_csv(ATAC_path + 'barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
cell_names['cell_ids'] = cell_names['cell_ids'].str.replace('.','-')
X = csr_matrix(mmread(ATAC_path + 'matrix.mtx').T)
peak_name = pd.read_csv(ATAC_path + 'features.tsv', sep = '\t',header=None,index_col=None)
peak_name.columns = ['peak_ids']
adata_ATAC  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
adata_ATAC.var_names_make_unique()

adata_ATAC.layers['counts'] = adata_ATAC.X.copy()
sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
sc.pp.log1p(adata_ATAC)
adata_ATAC.layers['log-norm'] = adata_ATAC.X.copy()
sc.pp.highly_variable_genes(
    adata_ATAC,
    flavor="seurat_v3",
    n_top_genes=30000,
    subset=False
)
adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable].copy()

start_time = time.time()
adata = sca.models.organize_multiome_anndatas(
    adatas = [[adata_RNA], [adata_ATAC]],    # a list of anndata objects per modality, RNA-seq always goes first
    layers = [['counts'], ['log-norm']], # if need to use data from .layers, if None use .X
)

sca.models.MultiVAE.setup_anndata(
    adata,
    rna_indices_end=4000,
)
model = sca.models.MultiVAE(
    adata,
    losses=['nb', 'mse'],
    loss_coefs={'kl': 1e-1,
               'integ': 3000,
               },
)
model.train()
model.get_latent_representation()
latent = adata.obsm['latent'].copy()
np.savetxt(arguments[2]+"/Multigrate.csv", latent, delimiter=',')

