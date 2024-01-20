from scipy.io import mmread
from scipy.sparse import csr_matrix
import scarches as sca
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import muon
import gdown
import os, sys, time
# import importlib
# importlib.reload(sca)

import warnings
warnings.filterwarnings("ignore")

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
    adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
    # adata_RNA.var_names_make_unique()
    sc.pp.filter_cells(adata_RNA, min_genes=1)
    sc.pp.filter_genes(adata_RNA, min_cells=20)

    cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids']
    X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
    peak_name = pd.read_csv(path+'/ATAC/peaks.tsv',header=None,index_col=None)
    peak_name.columns = ['peak_ids']
    adata_ATAC = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
    sc.pp.filter_cells(adata_ATAC, min_genes=1)
    sc.pp.filter_genes(adata_ATAC, min_cells=1)

    locals()['adata_paired'+str(i)] = ad.concat([adata_RNA, adata_ATAC], merge = "same",axis=1)
    eval('adata_paired'+str(i)).obs_keys
    modality = ['Gene Expression']*adata_RNA.shape[1]+['Peaks']*adata_ATAC.shape[1]
    locals()['adata_paired'+str(i)].var['modality'] = modality

del adata_RNA, adata_ATAC

adata = ad.concat([adata_paired1, adata_paired2],axis=0)
adata.obs['batch'] = ['batch1']*adata_paired1.n_obs + ['batch2']*adata_paired2.n_obs
modality = ['Gene Expression']*adata.shape[1]
for i in np.where(adata.var_names.str.contains("chr"))[0]:
    modality[i] = 'Peaks'
adata.var['modality']=modality
del adata_paired1, adata_paired2

rna = adata[:,adata.var['modality']=='Gene Expression']
rna.layers['counts'] = rna.X.copy()

sc.pp.highly_variable_genes(
    rna,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=False
)

sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
# subset to hvg
# sc.pp.highly_variable_genes(rna, n_top_genes=2990, batch_key='batch')
rna = rna[:, rna.var.highly_variable].copy()

# split again
rna_1 = rna[rna.obs['batch'] == 'batch1'].copy()
rna_2 = rna[rna.obs['batch'] == 'batch2'].copy()

atac = adata[:,adata.var['modality']=='Peaks']
atac.layers['counts'] = atac.X.copy()

atac.X = atac.layers['counts'].copy()

sc.pp.normalize_total(atac, target_sum=1e4)
sc.pp.log1p(atac)
atac.layers['log-norm'] = atac.X.copy()

# split again
atac_1 = atac[atac.obs['batch'] == 'batch1'].copy()
atac_2 = atac[atac.obs['batch'] == 'batch2'].copy()

adata = sca.models.organize_multiome_anndatas(
    adatas = [[rna_1, rna_2], [atac_1, atac_2]],    # a list of anndata objects per modality, RNA-seq always goes first
    layers = [['counts', 'counts'], ['log-norm', 'log-norm']], # if need to use data from .layers, if None use .X
)
del rna_1, rna_2, atac_1, atac_2

sca.models.MultiVAE.setup_anndata(
    adata,
    categorical_covariate_keys=['group'],
    rna_indices_end=2990,
)

model = sca.models.MultiVAE(
    adata,
    losses=['nb', 'mse'],
    loss_coefs={'kl': 1e-1,
               'integ': 3000,},
    integrate_on='group',
    mmd='marginal',
)
model.train()

model.get_latent_representation()
latent = adata.obsm['latent'].copy()

np.savetxt(save_path+dataset_id+"_latent_scArches.csv", latent, delimiter=',')
