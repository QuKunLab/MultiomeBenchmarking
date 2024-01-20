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

import sys
import warnings
warnings.filterwarnings("ignore")

dataset_id = "Dataset7"

save_path = "../results/Horizontal/RNA_Protein/"  ## path to save results

# read data
data_path = "../dataset/Horizontal/RNA_Protein/"+dataset_id ## path to raw data

## batch1
path = data_path+"/batch1"

cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_names = pd.read_csv(path+'/RNA/genes.tsv', sep = '\t',  header=None, index_col=None)
gene_names.columns =  ['gene_ids']
adata_batch1 = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
del X
adata_batch1.var_names_make_unique()

adata_ADT = sc.read_csv(path+"/ADT/ADT.csv").T
adata_batch1.obsm['protein_expression'] = adata_ADT.to_df()
protein_name = adata_ADT.var_names
del adata_ADT

## batch2
path = data_path+"/batch2"

cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_name = pd.read_csv(path+'/RNA/genes.tsv',header=None,index_col=None)
gene_name.columns = ['gene_ids']
adata_batch2 = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_name.gene_ids))
del X
adata_batch2.var_names_make_unique()

adata_ADT = sc.read_csv(path+"/ADT/ADT.csv").T
adata_batch2.obsm['protein_expression'] = adata_ADT.to_df()
protein_name = adata_ADT.var_names
del adata_ADT

adata = ad.concat([adata_batch1, adata_batch2],axis=0)
adata.obs['batch'] = ['batch1']*adata_batch1.shape[0]+['batch2']*adata_batch2.shape[0]
del adata_batch1, adata_batch2

sc.pp.highly_variable_genes(
    adata,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=True
)

sca.models.TOTALVI.setup_anndata(
    adata,
    batch_key="batch",
    protein_expression_obsm_key="protein_expression"
)
print(adata.shape)
arches_params = dict(use_layer_norm="both", use_batch_norm="none")

model = sca.models.TOTALVI(adata, **arches_params)

model.train()

latent = model.get_latent_representation()
np.savetxt(save_path+dataset_id+"_latent_scArches.csv", latent, delimiter=',')
