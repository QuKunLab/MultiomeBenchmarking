import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import muon as mu
from scipy.io import mmread
from scipy.sparse import csr_matrix
import os, sys, time
from muon import MuData
import importlib
# sys.path.append(".")
# sys.path.append("./mofapy2/")
import mofapy2

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
    locals()['adata_RNA_'+str(i)] = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
    # eval('adata_RNA_'+str(i)).var_names_make_unique()
    sc.pp.filter_cells(eval('adata_RNA_'+str(i)), min_genes=1)
    sc.pp.filter_genes(eval('adata_RNA_'+str(i)), min_cells=1)

    # peak information
    cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids']
    X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
    peak_name = pd.read_csv(path+'/ATAC/peaks.tsv',header=None,index_col=None)
    peak_name.columns = ['peak_ids']
    locals()['adata_ATAC_'+str(i)] = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
    sc.pp.filter_cells(eval('adata_ATAC_'+str(i)), min_genes=1)
    sc.pp.filter_genes(eval('adata_ATAC_'+str(i)), min_cells=1)

    locals()['n'+str(i)] = eval('adata_ATAC_'+str(i)).n_obs

del X

rna = ad.concat([adata_RNA_1, adata_RNA_2],axis=0)
rna.obs['batch'] = ['batch1']*adata_RNA_1.n_obs + ['batch2']*adata_RNA_2.n_obs
rna.layers['counts'] = rna.X.copy()
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
rna.layers['lognorm'] = rna.X.copy()
sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
sc.pp.scale(rna,max_value=10)


atac = ad.concat([adata_ATAC_1, adata_ATAC_2],axis=0)
atac.obs['batch'] = ['batch1']*adata_ATAC_1.n_obs + ['batch2']*adata_ATAC_2.n_obs
atac.layers["counts"] = atac.X.copy()
sc.pp.normalize_total(atac, target_sum=1e4)
sc.pp.log1p(atac)
atac.layers["lognorm"] = atac.X.copy()
sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
sc.pp.scale(atac, max_value=10)

n1 = adata_ATAC_1.n_obs
n2 = adata_ATAC_2.n_obs

del adata_RNA_1, adata_RNA_2, adata_ATAC_1, adata_ATAC_2

mdata = MuData({'rna': rna, 'atac': atac})
del rna, atac

mdata.obs['batch'] = ['batch1']*n1 + ['batch2']*n2

mu.tl.mofa(mdata,use_var=None, groups_label="batch", gpu_mode=False,save_data=False)
np.savetxt(save_path+dataset_id+"_latent_MOFA.csv", mdata.obsm['X_mofa'], delimiter=',')
