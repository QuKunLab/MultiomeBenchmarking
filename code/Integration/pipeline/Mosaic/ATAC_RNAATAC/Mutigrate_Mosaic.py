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


arguments = sys.argv
if not os.path.exists(arguments[1]):
    print("Input RNA+ATAC Data does not exist.")
if not os.path.exists(arguments[2]):
    print("Input ATAC Data does not exist.")
if not os.path.exists(arguments[3]):
    print("Don't input DataName.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")
print("Input file: "+ arguments[2] + " \n")
print("Data Name: "+ arguments[3] + " \n")

# 10x multiome
path = arguments[1]
# gene expression
cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_names = pd.read_csv(path+'/RNA/features.tsv', sep = '\t',  header=None, index_col=None) 
gene_names.columns =  ['gene_ids'] 
adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA.var_names_make_unique()
sc.pp.filter_cells(adata_RNA, min_genes=1)
sc.pp.filter_genes(adata_RNA, min_cells=20)

# peak information
cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
peak_name = pd.read_csv(path+'/ATAC/features.tsv',header=None,index_col=None)
peak_name.columns = ['peak_ids']
adata_ATAC  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
sc.pp.filter_cells(adata_ATAC, min_genes=1)
sc.pp.filter_genes(adata_ATAC, min_cells=1)

path = arguments[2]
# peak information
cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
peak_name = pd.read_csv(path+'/ATAC/features.tsv',header=None,index_col=None)
peak_name.columns = ['peak_ids']
adata_atac  = ad.AnnData(X, obs=pd.DataFrame(index=["batch2_"+ing for ing in cell_names.cell_ids]), var=pd.DataFrame(index = peak_name.peak_ids))
del X
sc.pp.filter_cells(adata_atac, min_genes=1)
sc.pp.filter_genes(adata_atac, min_cells=1)


sc.pp.highly_variable_genes(
    adata_RNA,
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=True)
adata_RNA.layers['counts'] = adata_RNA.X.copy()

if not os.path.exists(arguments[3]+"/"):
    os.makedirs(arguments[3]+"/")
    
Df = pd.DataFrame(adata_RNA.var_names)
Df.to_csv(arguments[3]+"/DEgene.csv", index=False,header=False)


atac = ad.concat([adata_ATAC, adata_atac])
atac.obs['batch'] = ['batch1']*adata_ATAC.n_obs + ['batch2']*adata_atac.n_obs

# sc.pp.highly_variable_genes(
#     atac,
#     batch_key="batch",
#     flavor="seurat_v3",
#     n_top_genes=30000,
#     subset=False)
adata_ATAC = atac[atac.obs['batch'] == 'batch1'].copy()
adata_atac = atac[atac.obs['batch'] == 'batch2'].copy()

sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
sc.pp.log1p(adata_ATAC)
# adata_ATAC = adata_ATAC[:,atac.var.highly_variable].copy()
adata_ATAC.layers['log-norm'] = adata_ATAC.X.copy()

# Df = pd.DataFrame(adata_ATAC.var_names)
# Df.to_csv(arguments[3]+"/DEpeak.csv", index=False,header=False)

sc.pp.normalize_total(adata_atac, target_sum=1e4)
sc.pp.log1p(adata_atac)
#adata_atac = adata_atac[:,atac.var.highly_variable].copy()
adata_atac.layers['log-norm'] = adata_atac.X.copy()

adata = sca.models.organize_multiome_anndatas(
    adatas = [[adata_RNA, None], [adata_ATAC, adata_atac]],    # a list of anndata objects per modality, RNA-seq always goes first
    layers = [['counts', None], ['log-norm', 'log-norm']], # if need to use data from .layers, if None use .X
)

sca.models.MultiVAE.setup_anndata(
    adata,
    categorical_covariate_keys=['group'],
    rna_indices_end=4000,
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
np.savetxt(arguments[3]+"/Multigrate.csv", latent, delimiter=',')
