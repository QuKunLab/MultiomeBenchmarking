from scipy.io import mmread
from scipy.sparse import csr_matrix
import anndata as ad
import pandas as pd
import scvi
import numpy as np
import scanpy as sc
scvi.settings.seed = 420
import os, sys, time

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

    # peak information
    cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids']
    X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
    peak_name = pd.read_csv(path+'/ATAC/peaks.tsv',header=None,index_col=None)
    peak_name.columns = ['peak_ids']
    locals()['adata_ATAC'+str(i)]  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))

del X

adata_RNA = ad.concat([adata_RNA1,adata_RNA2])
adata_RNA.obs['batch'] = ['batch1']*adata_RNA1.n_obs + ['batch2']*adata_RNA2.n_obs
sc.pp.highly_variable_genes(
    adata_RNA,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=True
)
adata_RNA1 = adata_RNA[adata_RNA.obs['batch'] == 'batch1'].copy()
adata_RNA2 = adata_RNA[adata_RNA.obs['batch'] == 'batch2'].copy()

for i in range(1,3):
    locals()['adata_paired'+str(i)] = ad.concat([eval('adata_RNA'+str(i)), eval('adata_ATAC'+str(i))], merge = "same",axis=1)
    eval('adata_paired'+str(i)).obs_keys
    modality = ['Gene Expression']*eval('adata_RNA'+str(i)).shape[1]+['Peaks']*eval('adata_ATAC'+str(i)).shape[1]
    locals()['adata_paired'+str(i)].var['modality']=modality

adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired1, adata_paired2)
del adata_paired1, adata_paired2

adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()

# print(adata_mvi.shape)
# # sc.pp.filter_genes(adata_mvi, min_cells=int(adata_mvi.shape[0] * 0.001))
# print(adata_mvi.shape)
scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")

model = scvi.model.MULTIVI(
    adata_mvi,
    n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
    n_regions=(adata_mvi.var["modality"] == "Peaks").sum(),
)
model.view_anndata_setup()

model.train()

latent = model.get_latent_representation()
np.savetxt(save_path+dataset_id+"_latent_MultiVI.csv", latent, delimiter=',')
