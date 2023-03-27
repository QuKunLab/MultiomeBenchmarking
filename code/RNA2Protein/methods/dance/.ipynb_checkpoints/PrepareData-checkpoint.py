import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.io import mmread, mmwrite
import os
import numpy as np

def read_RNA_ATAC(RNA_path, ATAC_path):
    # gene expression
    cell_names = pd.read_csv(RNA_path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids'] 
    X = csr_matrix(mmread(RNA_path+'matrix.mtx').T)
    gene_names = pd.read_csv(RNA_path+'genes.tsv', sep = '\t',  header=None, index_col=None) 
    gene_names.columns =  ['gene_ids'] 
    adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
    adata_RNA.var_names_make_unique()
    if ATAC_path is None:
        return adata_RNA
    # peak information
    cell_names = pd.read_csv(ATAC_path + 'barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids'] 
    X = csr_matrix(mmread(ATAC_path + 'matrix.mtx').T)
    peak_name = pd.read_csv(ATAC_path + 'peaks.bed',header=None,index_col=None)
    peak_name.columns = ['peak_ids']
    adata_ATAC  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
    return adata_RNA, adata_ATAC

data_path = "../Code2Provide/dataset/"
train_id = "21"
test_id = "22"
## Path to data
train_data_path = data_path + "dataset" + train_id + "_adata.h5ad"
test_data_path = data_path + "dataset" + test_id + "_adata.h5ad"
## Load training data
data = ad.read_h5ad(train_data_path)
X = csr_matrix(data.X)
cell_names = pd.DataFrame(data.obs_names)
cell_names.columns = ['cell_ids']
gene_names = pd.DataFrame(data.var_names)
gene_names.columns = ['gene_ids']
input_train_mod1 = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
# input_train_mod1.var['gene_ids'] = input_train_mod1.var_names
input_train_mod1.var['feature_types'] = pd.Categorical(len(input_train_mod1.var_names)*['GEX'])
input_train_mod1.obs['batch'] = pd.Categorical(len(input_train_mod1.obs)*['batch1'])
input_train_mod1.uns = {"dataset_id": train_id, "organism": "human"}
input_train_mod1.layers['counts'] = input_train_mod1.X.copy()

temp = data.obsm['protein_expression'].values
temp = csr_matrix(temp)
obs_mod2 = pd.DataFrame(index = data.obsm['protein_expression'].index)
var_mod2 = pd.DataFrame(index = data.obsm['protein_expression'].columns)
input_train_mod2 = ad.AnnData(temp, obs=obs_mod2, var=var_mod2)
# input_train_mod2.var['gene_ids'] = input_train_mod2.var_names
input_train_mod2.var['feature_types'] = pd.Categorical(len(input_train_mod2.var_names)*['ADT'])
input_train_mod2.obs['batch'] = pd.Categorical(len(input_train_mod2.obs)*['batch1'])
input_train_mod2.uns = {"dataset_id": train_id, "organism": "human"}
input_train_mod2.layers['counts'] = input_train_mod2.X.copy()

## Load test data
test_data = ad.read_h5ad(test_data_path)
X = csr_matrix(test_data.X)
cell_names = pd.DataFrame(test_data.obs_names)
cell_names.columns = ['cell_ids']
gene_names = pd.DataFrame(test_data.var_names)
gene_names.columns = ['gene_ids']
input_test_mod1 = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
# input_test_mod1.var['gene_ids'] = pd.Categorical(gene_names.index)
input_test_mod1.var['feature_types'] = pd.Categorical(len(input_test_mod1.var_names)*['GEX'])
input_test_mod1.obs['batch'] = pd.Categorical(len(input_test_mod1.obs)*['batch2'])
input_test_mod1.uns = {"dataset_id": test_id, "organism": "human"}
input_test_mod1.layers['counts'] = input_test_mod1.X.copy()

temp = test_data.obsm['protein_expression'].values
temp = csr_matrix(temp)
obs_mod2 = pd.DataFrame(index = test_data.obsm['protein_expression'].index)
var_mod2 = pd.DataFrame(index = test_data.obsm['protein_expression'].columns)
input_test_mod2 = ad.AnnData(temp, obs=obs_mod2, var=var_mod2)
# input_train_mod2.var['gene_ids'] = input_train_mod2.var_names
input_test_mod2.var['feature_types'] = pd.Categorical(len(input_test_mod2.var_names)*['ADT'])
input_test_mod2.obs['batch'] = pd.Categorical(len(input_test_mod2.obs)*['batch2'])
input_test_mod2.uns = {"dataset_id": test_id, "organism": "human"}
input_test_mod2.layers['counts'] = input_test_mod2.X.copy()

## Preprocess data
gene_idx_train = np.array(input_train_mod1.X.sum(axis=0)>0).flatten()
gene_idx_test = np.array(input_test_mod1.X.sum(axis=0)>0).flatten()
input_test_mod1 = input_test_mod1[:,gene_idx_test]
input_train_mod1 = input_train_mod1[:,gene_idx_train]

input_train_mod1.layers['counts'] = input_train_mod1.X.copy()
input_test_mod1.layers['counts'] = input_test_mod1.X.copy()

genes_inter = input_train_mod1.var_names.intersection(input_test_mod1.var_names)
input_train_mod1 = input_train_mod1[:,genes_inter].copy()
input_test_mod1 = input_test_mod1[:,genes_inter].copy()

## Store data to the specific location
os.chdir("../dance/data/")

input_train_mod1.write_h5ad("openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod1.h5ad", compression = "gzip")
input_train_mod2.write_h5ad("openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod2.h5ad", compression = "gzip")
input_test_mod1.write_h5ad("openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_test_mod1.h5ad", compression = "gzip")
input_test_mod2.write_h5ad("openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_test_mod2.h5ad", compression = "gzip")

os.system('zip -r openproblems_bmmc_cite_phase2_rna.zip openproblems_bmmc_cite_phase2_rna')