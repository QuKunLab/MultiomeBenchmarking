from scipy.io import mmread
from scipy.sparse import csr_matrix
import scarches as sca
import scanpy as sc
# import multigrate as mtg
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
    print("Input RNA Data does not exist.")

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
atac  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
sc.pp.filter_cells(atac, min_genes=1)
sc.pp.filter_genes(atac, min_cells=1)


path = arguments[2]
# gene expression
cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_names = pd.read_csv(path+'/RNA/features.tsv', sep = '\t',  header=None, index_col=None) 
gene_names.columns =  ['gene_ids'] 
adata_rna = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_rna.var_names_make_unique()
sc.pp.filter_cells(adata_rna, min_genes=1)
sc.pp.filter_genes(adata_rna, min_cells=20)

rna = ad.concat([adata_RNA, adata_rna],axis=0)
rna.obs_names_make_unique()
rna.obs['batch'] = ['batch1']*adata_RNA.n_obs + ['batch2']*adata_rna.n_obs

del adata_RNA, adata_rna
 
sc.pp.highly_variable_genes(
    rna,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=True
)
rna.layers['counts'] = rna.X.copy()

if not os.path.exists(arguments[3]+"/"):
    os.makedirs(arguments[3]+"/")

Df = pd.DataFrame(rna.var_names)
Df.to_csv(arguments[3]+"/DEgene.csv", index=False,header=False)

# split again
rna_1 = rna[rna.obs['batch'] == 'batch1'].copy()
rna_2 = rna[rna.obs['batch'] == 'batch2'].copy()

#sc.pp.highly_variable_genes(
#    atac,
#    flavor="seurat_v3",
#    n_top_genes=30000,
#    subset=False
#)
sc.pp.normalize_total(atac, target_sum=1e4)
sc.pp.log1p(atac)
atac.layers['log-norm'] = atac.X.copy()
#atac = atac[:, atac.var.highly_variable].copy()

#Df = pd.DataFrame(atac.var_names)
#Df.to_csv(arguments[3]+"/DEpeak.csv", index=False,header=False)

# start_time = time.time() #From now on time
atac.obs_names = rna_1.obs_names
adata = sca.models.organize_multiome_anndatas(
    adatas = [[rna_1, rna_2], [atac, None]],    # a list of anndata objects per modality, RNA-seq always goes first
    layers = [['counts', 'counts'], ['log-norm', None]], # if need to use data from .layers, if None use .X
)
del rna_1, rna_2, atac

sca.models.MultiVAE.setup_anndata(
    adata,
    categorical_covariate_keys=['batch'],
    rna_indices_end=4000,
)

model = sca.models.MultiVAE(
    adata,
    losses=['nb', 'mse'],
    loss_coefs={'kl': 1e-1,
               'integ': 3000,
               },
    integrate_on='batch',
    mmd='marginal',
)

model.train()

model.get_latent_representation()
latent = adata.obsm['latent'].copy()
np.savetxt(arguments[3]+"/Multigrate.csv", latent, delimiter=',')
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"running time:{execution_time} s")
