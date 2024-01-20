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

# read RNA data
data_path = "../dataset/Horizontal/RNA_Protein/"+dataset_id ## path to raw data

## RNA
RNA_path = data_path+"/batch1/RNA/"
cell_names = pd.read_csv(RNA_path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(RNA_path+'matrix.mtx').T)
gene_names = pd.read_csv(RNA_path+'genes.tsv', sep = '\t',  header=None, index_col=None)
gene_names.columns =  ['gene_ids']
adata_RNA_1 = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA_1.var_names_make_unique()
adata_RNA_1.obs['batch'] = 'batch1'
adata_RNA_1.layers["counts"] = adata_RNA_1.X.copy()

RNA_path = data_path+"/batch2/RNA/"
cell_names = pd.read_csv(RNA_path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(RNA_path+'matrix.mtx').T)
gene_names = pd.read_csv(RNA_path+'genes.tsv', sep = '\t',  header=None, index_col=None)
gene_names.columns =  ['gene_ids']
adata_RNA_2 = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA_2.var_names_make_unique()
adata_RNA_2.obs['batch'] = 'batch2'
adata_RNA_2.layers["counts"] = adata_RNA_2.X.copy()
# concat
rna = ad.concat([adata_RNA_1, adata_RNA_2])
rna.obs['batch'] = ['batch1']*adata_RNA_1.n_obs + ['batch2']*adata_RNA_2.n_obs
rna.layers['counts'] = rna.X.copy()

sc.pp.highly_variable_genes(
    rna,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=True
)

rna = rna[:, rna.var.highly_variable].copy()
# split again
adata_RNA_1 = rna[rna.obs['batch'] == 'batch1'].copy()
adata_RNA_2 = rna[rna.obs['batch'] == 'batch2'].copy()
adata_RNA_1.shape, adata_RNA_2.shape

### ADT
ADT_path = data_path+"/batch1/ADT/"
adata_ADT = pd.read_csv(ADT_path+'ADT.csv',index_col = 0)
adata_ADT.columns = adata_ADT.columns.str.replace('.','-')
adata_ADT.index = adata_ADT.index.str.replace('.','_')
adata_ADT.index = adata_ADT.index.str.replace('-','_')
adata_ADT_1 = ad.AnnData(adata_ADT.T)

adata_ADT_1.X = adata_ADT_1.X.astype(np.float64)
adata_ADT_1.layers["counts"] = adata_ADT_1.X.copy()
muon.prot.pp.clr(adata_ADT_1)
adata_ADT_1.layers['clr'] = adata_ADT_1.X.copy()
adata_ADT_1.obs['batch'] = 'batch1'

ADT_path = data_path+"/batch2/ADT/"
adata_ADT = pd.read_csv(ADT_path+'ADT.csv',index_col = 0)
adata_ADT.columns = adata_ADT.columns.str.replace('.','-')
adata_ADT.index = adata_ADT.index.str.replace('.','_')
adata_ADT.index = adata_ADT.index.str.replace('-','_')
adata_ADT_2 = ad.AnnData(adata_ADT.T)

adata_ADT_2.X = adata_ADT_2.X.astype(np.float64)
adata_ADT_2.layers["counts"] = adata_ADT_2.X.copy()
muon.prot.pp.clr(adata_ADT_2)
adata_ADT_2.layers['clr'] = adata_ADT_2.X.copy()
adata_ADT_2.obs['batch'] = 'batch2'

adata = sca.models.organize_multiome_anndatas(
    adatas = [[adata_RNA_1,adata_RNA_2], [adata_ADT_1,adata_ADT_2]],    # a list of anndata objects per modality, RNA-seq always goes first
    layers = [['counts','counts'], ['clr','clr']], # if need to use data from .layers, if None use .X
)
adata

sca.models.MultiVAE.setup_anndata(
    adata,
    categorical_covariate_keys=['batch'],
    rna_indices_end=4000,
)

model = sca.models.MultiVAE(
    adata,
    # losses=['nb', 'mse', 'mse'],
    losses=['nb', 'mse'],

    loss_coefs={'kl': 1e-1,
               'integ': 3000,
               },
    integrate_on='batch',
    mmd='marginal',
)

model.train()

model.get_latent_representation()
adata.obsm['latent_ref'] = adata.obsm['latent'].copy()

np.savetxt(results_path+dataset_id+"_latent_Multigrate.csv", adata.obsm['latent'], delimiter=',')
