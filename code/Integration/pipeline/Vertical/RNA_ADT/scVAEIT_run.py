import os
import sys
import time
import numpy as np
import pandas as pd
import anndata as ad
import scipy as sp
import scipy.sparse
import h5py

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix,lil_matrix
from scipy.io import mmread, mmwrite
import scanpy as sc

from functools import partial
sys.path.append("../../")
from scVAEIT.VAEIT import scVAEIT

arguments = sys.argv

if not os.path.exists(arguments[1]):
    print("Input Data does not exist.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")

path = arguments[1]

RNA_path = path+"/RNA/"
cell_names = pd.read_csv(RNA_path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
cell_names['cell_ids'] = cell_names['cell_ids'].str.replace('.','-')
X = csr_matrix(mmread(RNA_path+'matrix.mtx').T)
gene_names = pd.read_csv(RNA_path+'features.tsv', sep = '\t',  header=None, index_col=None) 
gene_names.columns =  ['gene_ids'] 
adata_rna = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_rna.var_names_make_unique()
adata_rna.obs["data_type"] = "rna"
del X

adata_rna.uns['counts'] = lil_matrix(adata_rna.X)
# filter genes
sc.pp.filter_genes(adata_rna, min_cells=1)

sc.pp.highly_variable_genes(
    adata_rna,
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=False
)
# CMP
sc.pp.normalize_total(adata_rna, target_sum=1e6)
# log transformation
sc.pp.log1p(adata_rna)
adata_rna = adata_rna[:, adata_rna.var.highly_variable].copy()

gene_list = pd.DataFrame(data = list(adata_rna.var_names),columns = ['gene_id'])
gene_list.to_csv(arguments[2]+"/GeneList.csv")

ADT_raw = pd.read_csv(path +"/ADT.csv", index_col=0, header=0).T
X_raw = adata_rna.X

gene_names_raw = adata_rna.var_names.values.astype(str)
ADT_names_raw = ADT_raw.keys().to_numpy().astype(str)
Y_raw = ADT_raw.values
del ADT_raw

X = X_raw

gene_names = gene_names_raw[np.array(np.sum(X, axis=0)).flatten()>10]
X = X[:,np.array(np.sum(X, axis=0)).flatten()>10]

Y = Y_raw
ADT_names = ADT_names_raw[np.array(np.sum(Y, axis=0)).flatten()>10]
Y = Y[:,np.array(np.sum(Y, axis=0)).flatten()>10]

gene_names = gene_names.astype(str)

# 调整每一行的值以使其和为10000
X = X.toarray()
scaling_factors = 10000 / X.sum(axis=1)
X = np.array(X) * scaling_factors[:, np.newaxis]
X = np.log(np.matrix(X+1.))

# 调整每一行的值以使其和为10000
scaling_factors = 10000 / np.array(Y.sum(axis=1))
Y = np.array(Y) * scaling_factors[:, np.newaxis]
Y = np.log(np.matrix(Y+1.))

data = np.c_[X, Y]
del X, Y
data_train_norm = np.exp(data)-1

path_root = './scVAEIT_res'

dim_input_arr = [len(gene_names), len(ADT_names)]
config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256], 
    'dim_latent':32,
    'dim_block': np.array(dim_input_arr),
    'dist_block':['NB','NB'], 
    'dim_block_enc':np.array([256, 128]),
    'dim_block_dec':np.array([256, 128]),
    'dim_block_embed':np.array([32, 16]),
    
    'block_names':np.array(['rna', 'adt']),
    'uni_block_names':np.array(['rna','adt']),
    
    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.1,0.9]),
    'beta_reverse':0.5,

    "p_feat" : 0.2,
    "p_modal" : np.ones(2)/2,
    
}
config = SimpleNamespace(**config)
n_samples = 50

model = scVAEIT(config, data)

model.train(
    valid=False, num_epoch=500, batch_size=512, save_every_epoch=50,
    verbose=False, checkpoint_dir=path_root+'checkpoint/'
)

latent =  model.get_latent_z()
np.savetxt(arguments[2]+"/scVAEIT.csv", latent, delimiter=',')



