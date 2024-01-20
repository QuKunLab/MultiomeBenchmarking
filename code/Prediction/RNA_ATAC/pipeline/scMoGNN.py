import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.io import mmread, mmwrite
import os
import numpy as np
tmp_path = './dance/data/openproblems_bmmc_multiome_phase2_rna/'

cell_names = pd.read_csv('./Dataset37/Train/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns = ['cell_ids'] 
X = csr_matrix(mmread('./Dataset37/Train/RNA/matrix.mtx').T)
gene_names = pd.read_csv('./Dataset37/Train/RNA/features.tsv', sep = '\t', header=None, index_col=None) 
gene_names.columns = ['gene_ids'] 
input_train_mod1 = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
input_train_mod1.var_names_make_unique()

cell_names = pd.read_csv('./Dataset37/Test/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns = ['cell_ids'] 
X = csr_matrix(mmread('./Dataset37/Test/RNA/matrix.mtx').T)
gene_names = pd.read_csv('./Dataset37/Train/RNA/features.tsv', sep = '\t', header=None, index_col=None) 
gene_names.columns = ['gene_ids'] 
input_test_mod1 = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
input_test_mod1.var_names_make_unique()


cell_names = pd.read_csv('./Dataset37/Train/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns = ['cell_ids'] 
X = csr_matrix(mmread('./Dataset37/Train/ATAC/matrix.mtx').T)
gene_names = pd.read_csv('./Dataset37/Train/ATAC/features.tsv', sep = '\t', header=None, index_col=None) 
gene_names.columns = ['gene_ids'] 
input_train_mod2 = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
input_train_mod2.var_names_make_unique()

cell_names = pd.read_csv('./Dataset37/Test/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns = ['cell_ids'] 
X = csr_matrix(mmread('./Dataset37/Test/ATAC/matrix.mtx').T)
gene_names = pd.read_csv('./Dataset37/Train/ATAC/features.tsv', sep = '\t', header=None, index_col=None) 
gene_names.columns = ['gene_ids'] 
input_test_mod2 = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
input_test_mod2.var_names_make_unique()

input_train_mod1.var['feature_types'] = pd.Categorical(len(input_train_mod1.var_names)*['GEX'])
input_train_mod1.obs['batch'] = pd.Categorical(len(input_train_mod1.obs)*['batch1'])
input_train_mod1.uns = {'dataset_id': 'human_pbmc_3k', 'organism': 'human'}
input_train_mod1.layers['counts'] = input_train_mod1.X.copy()
input_train_mod2.var['feature_types'] = pd.Categorical(len(input_train_mod2.var_names)*['ATAC'])
input_train_mod2.obs['batch'] = pd.Categorical(len(input_train_mod2.obs)*['batch1'])
input_train_mod2.uns = {'dataset_id': 'human_pbmc_3k', 'organism': 'human'}
input_train_mod2.layers['counts'] = input_train_mod2.X.copy()
input_test_mod1.var['feature_types'] = pd.Categorical(len(input_test_mod1.var_names)*['GEX'])
input_test_mod1.obs['batch'] = pd.Categorical(len(input_test_mod1.obs)*['batch1'])
input_test_mod1.uns = {'dataset_id': 'human_pbmc_3k', 'organism': 'human'}
input_train_mod1.layers['counts'] = input_train_mod1.X.copy()
input_test_mod2.var['feature_types'] = pd.Categorical(len(input_test_mod2.var_names)*['ATAC'])
input_test_mod2.obs['batch'] = pd.Categorical(len(input_test_mod2.obs)*['batch1'])
input_test_mod2.uns = {'dataset_id': 'human_pbmc_3k', 'organism': 'human'}
input_train_mod2.layers['counts'] = input_train_mod2.X.copy()
input_train_mod1.write_h5ad(tmp_path + "openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod1.h5ad", compression = None)
input_train_mod2.write_h5ad(tmp_path + "openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod2.h5ad", compression = None)
input_test_mod1.write_h5ad(tmp_path + "openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_test_mod1.h5ad", compression = None)
input_test_mod2.write_h5ad(tmp_path + "openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_test_mod2.h5ad", compression = None)
os.chdir("./dance")
datasets = 'scMoGNN'
os.system('python ./dance/examples/multi_modality/predict_modality/scmogcn.py --subtask openproblems_bmmc_multiome_phase2_rna --device cuda --dataset ' + f'{datasets}')