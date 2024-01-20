import os, sys, time
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
import h5py
import tensorflow as tf
import tensorflow_probability as tfp
import scanpy as sc
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

from types import SimpleNamespace

from scipy.sparse import csr_matrix, coo_matrix, csc_matrix,lil_matrix
from scipy.io import mmread, mmwrite

sys.path.append("../../")
from scVAEIT.VAEIT import scVAEIT
arguments = sys.argv
if not os.path.exists(arguments[1]):
    print("Input RNA+Protein Data does not exist.")
if not os.path.exists(arguments[2]):
    print("Input RNA Data does not exist.")
if not os.path.exists(arguments[3]):
    print("Don't input DataName.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")
print("Input file: "+ arguments[2] + " \n")
print("Data Name: "+ arguments[3] + " \n")


## 10x multiome
path = arguments[1]
# gene expression
cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_names = pd.read_csv(path+'/RNA/features.tsv', sep = '\t',  header=None, index_col=None) 
gene_names.columns =  ['gene_ids'] 
adata_RNA = sc.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
del X
adata_RNA.var_names_make_unique()
sc.pp.filter_cells(adata_RNA, min_genes=1)
sc.pp.filter_genes(adata_RNA, min_cells=20)

# peak information
adata_ADT = sc.read_csv(path+"/ADT.csv").T
sc.pp.filter_cells(adata_ADT, min_genes=1)
sc.pp.filter_genes(adata_ADT, min_cells=1)

path = arguments[2]
## scRNA-seq
cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_name = pd.read_csv(path+'/RNA/features.tsv',header=None,index_col=None)
gene_name.columns = ['gene_ids']
adata_rna  = sc.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_name.gene_ids))
del X
sc.pp.filter_cells(adata_rna, min_genes=1)
sc.pp.filter_genes(adata_rna, min_cells=20)


batches = [0] * adata_RNA.shape[0] + [1] * adata_rna.shape[0]
batches = np.array(batches)
batches = batches.reshape((batches.shape[0],1))

overlap_gene = set(adata_rna.var_names).intersection(adata_RNA.var_names)
adata_RNA = adata_RNA[:,list(overlap_gene)]
adata_rna = adata_rna[:,adata_RNA.var_names]

# start_time = time.time()
adata = sc.concat([adata_RNA, adata_rna], axis=0)
adata.obs['batch'] = ['batch1']*adata_RNA.n_obs + ['batch2']*adata_rna.n_obs
ncell_rna = adata_rna.shape[0]
del adata_RNA, adata_rna

sc.pp.highly_variable_genes(
    adata,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=False)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata = adata[:, adata.var.highly_variable].copy()

sc.pp.normalize_total(adata_ADT, target_sum=1e4)
sc.pp.log1p(adata_ADT)

dim_input_arr = np.array([len(adata.var_names),len(adata_ADT.var_names)])
print(dim_input_arr)

X = adata.X.toarray()
Y = np.vstack((adata_ADT.X, np.zeros((ncell_rna, adata_ADT.shape[1]))))
del adata_ADT, adata

data = np.c_[X, Y]
del X, Y

id_X_Batch1 = np.array(range(dim_input_arr[0]), dtype=np.int32)
id_Y_Batch1 = np.array(range(dim_input_arr[1]), dtype=np.int32)  

masks = - np.ones((2, np.sum(dim_input_arr)), dtype=np.float32)
masks[0,id_X_Batch1] = 0.
masks[0,id_Y_Batch1+dim_input_arr[0]] = 0.
masks[1,id_X_Batch1] = 0.

masks = tf.convert_to_tensor(masks, dtype=tf.float32)
path_root = './scVAEIT_res/'

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
    'beta_modal':np.array([0.15,0.85]),
    'beta_reverse':0.5,

    "p_feat" : 0.2,
    "p_modal" : np.ones(2)/2,
}

config = SimpleNamespace(**config)


model = scVAEIT(config, data, masks, batches)
del data, masks
model.train(valid=False, num_epoch=300, batch_size=64, save_every_epoch=300,
        verbose=False, checkpoint_dir=path_root+'checkpoint/')

latent =  model.get_latent_z()
np.savetxt(arguments[3]+"/scVAEIT.csv", latent, delimiter=',')

# np.savetxt("laten.csv", model.get_latent_z(), delimiter=',')
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"running time:{execution_time} s")
