import scanpy as sc
import anndata as ad
import sys
import torch
import numpy as np
import scarches as sca
import matplotlib.pyplot as plt
import numpy as np
import scvi as scv
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix
import gdown
import json
import time
import os

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

# read ADT data
adata_ADT = pd.read_csv(data_path+'/ADT.csv',index_col = 0)
adata_ADT.columns = adata_ADT.columns.str.replace('.','-')
adata_ADT.index = adata_ADT.index.str.replace('.','_')
adata_ADT.index = adata_ADT.index.str.replace('-','_')
adata_RNA.obsm["protein_expression"] = adata_ADT.T
modality = ['batch1']*adata_RNA.shape[0]
adata_RNA.obs['batch']=modality

sca.models.TOTALVI.setup_anndata(
    adata_RNA,
    batch_key="batch",
    protein_expression_obsm_key="protein_expression"
)
arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
)
vae_ref = sca.models.TOTALVI(
    adata_RNA,
    **arches_params
)
vae_ref.train()

adata_RNA.obsm["X_scArches"] = vae_ref.get_latent_representation()
latent = adata_RNA.obsm['X_scArches'].copy()
np.savetxt(arguments[2]+"/scArches.csv", latent, delimiter=',')
