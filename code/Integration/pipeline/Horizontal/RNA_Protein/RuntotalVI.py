import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
from scipy.io import mmread
from scipy.sparse import csr_matrix
import scanpy as sc
import scvi

dataset_id = "Dataset7"

save_path = "../results/Horizontal/RNA_Protein/"  ## path to save results

# read data
data_path = "../dataset/Horizontal/RNA_Protein/"+dataset_id ## path to raw data

## batch1
RNA_path = data_path+"/batch1/RNA/"
cell_names = pd.read_csv(RNA_path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(RNA_path+'matrix.mtx').T)
gene_names = pd.read_csv(RNA_path+'genes.tsv', sep = '\t',  header=None, index_col=None)
gene_names.columns =  ['gene_ids']
adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA.var_names_make_unique()

ADT_path = data_path+"/batch1/ADT/"
adata_ADT = pd.read_csv(ADT_path+'ADT.csv',index_col = 0)
adata_ADT.columns = adata_ADT.columns.str.replace('.','-')
adata_ADT.index = adata_ADT.index.str.replace('.','_')
adata_ADT.index = adata_ADT.index.str.replace('-','_')
adata_ADT = ad.AnnData(adata_ADT.T)

adata_batch1 = adata_RNA
adata_batch1.obsm['protein_expression'] = adata_ADT.to_df()

## batch2
RNA_path = data_path+"/batch2/RNA/"
cell_names = pd.read_csv(RNA_path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(RNA_path+'matrix.mtx').T)
gene_names = pd.read_csv(RNA_path+'genes.tsv', sep = '\t',  header=None, index_col=None)
gene_names.columns =  ['gene_ids']
adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA.var_names_make_unique()

ADT_path = data_path+"/batch2/ADT/"
adata_ADT = pd.read_csv(ADT_path+'ADT.csv',index_col = 0)
adata_ADT.columns = adata_ADT.columns.str.replace('.','-')
adata_ADT.index = adata_ADT.index.str.replace('.','_')
adata_ADT.index = adata_ADT.index.str.replace('-','_')
adata_ADT = ad.AnnData(adata_ADT.T)

adata_batch2 = adata_RNA
adata_batch2.obsm['protein_expression'] = adata_ADT.to_df()

adata = ad.concat([adata_batch1, adata_batch2],axis=0)
adata.obs_keys

modality = ['batch1']*adata_batch1.shape[0]+['batch2']*adata_batch2.shape[0]
adata.obs['batch']=modality
batch = np.array(modality)

sc.pp.highly_variable_genes(
    adata,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=True
)

# Intergration

## setup TotalVI model
scvi.model.TOTALVI.setup_anndata(adata, batch_key="batch", protein_expression_obsm_key="protein_expression")

model = scvi.model.TOTALVI(
    adata,
    latent_distribution="normal",
    n_layers_decoder=2
)

#model.view_anndata_setup()

model.train()

adata.obsm["X_totalVI"] = model.get_latent_representation()
np.savetxt(save_path+dataset_id+"_latent_totalVI.csv", adata.obsm["X_totalVI"], delimiter=',')

