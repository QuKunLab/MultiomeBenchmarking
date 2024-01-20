import os
import sys, time
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
import h5py
import tensorflow as tf
import tensorflow_probability as tfp
import scanpy as sc
sys.path.append("../pipeline/scVAEIT/")
from scVAEIT.VAEIT import scVAEIT

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

from types import SimpleNamespace

from scipy.sparse import csr_matrix, coo_matrix, csc_matrix,lil_matrix
from scipy.io import mmread, mmwrite

dataset_id = "Dataset7"

save_path = "../results/Horizontal/RNA_Protein/"  ## path to save results
data_path = "../dataset/Horizontal/RNA_Protein/"+dataset_id ## path to raw data

# gene expression
path = data_path+"/batch1/RNA/"
cell_names = pd.read_csv(path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(path+'matrix.mtx').T)
gene_names = pd.read_csv(path+'genes.tsv', sep = '\t',  header=None, index_col=None)
gene_names.columns =  ['gene_ids']
adata_RNA_1 = sc.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA_1.var_names_make_unique()
sc.pp.filter_cells(adata_RNA_1, min_genes=1)
sc.pp.filter_genes(adata_RNA_1, min_cells=1)

adata_adt_1 = sc.read_csv(data_path+"/batch1/ADT/ADT.csv").T

# gene expression
path = data_path+"/batch2/RNA/"
cell_names = pd.read_csv(path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(path+'matrix.mtx').T)
gene_names = pd.read_csv(path+'genes.tsv', sep = '\t',  header=None, index_col=None)
gene_names.columns =  ['gene_ids']
adata_RNA_2 = sc.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA_2.var_names_make_unique()
sc.pp.filter_cells(adata_RNA_2, min_genes=1)
sc.pp.filter_genes(adata_RNA_2, min_cells=1)

adata_adt_2 = sc.read_csv(data_path+"/batch2/ADT/ADT.csv").T

batches = [1] * adata_RNA_1.shape[0] + [2] * adata_RNA_2.shape[0]
batches = np.array(batches)
batches = batches.reshape((batches.shape[0],1))

adata = sc.concat([adata_RNA_1, adata_RNA_2], axis=0)
ncell_RNA_1 = adata_RNA_1.shape[0]
ncell_RNA_2 = adata_RNA_2.shape[0]
adata.obs['batch'] = ['batch1']*ncell_RNA_1 + ['batch2']*ncell_RNA_2
del adata_RNA_1,adata_RNA_2

sc.pp.highly_variable_genes(
    adata,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=False
)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata = adata[:, adata.var.highly_variable].copy()
X = adata.X.toarray()
# del adata


adata_adt = sc.concat([adata_adt_1,adata_adt_2])
sc.pp.normalize_total(adata_adt, target_sum=1e4)
sc.pp.log1p(adata_adt)
ADT_names = adata_adt.var_names
gene_names = adata.var_names
dim_input_arr = np.array([len(adata.var_names),len(adata_adt.var_names)])
print(dim_input_arr)

# Y = np.vstack((adata_adt_0.X, adata_adt_1.X))
Y = adata_adt.X
del adata_adt
data = np.c_[X, Y]
del X, Y

id_X_Batch1 = np.array(range(dim_input_arr[0]), dtype=np.int32)
id_Y_Batch1 = np.array(range(dim_input_arr[1]), dtype=np.int32)
id_X_Batch2 = np.array(range(dim_input_arr[0]), dtype=np.int32)
id_Y_Batch2 = np.array(range(dim_input_arr[1]), dtype=np.int32)
# masks = np.zeros((2, np.sum(dim_input_arr)), dtype=np.float32)

# masks = tf.convert_to_tensor(masks, dtype=tf.float32)
path_root = './scVAEIT_res/'

config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256],
    'dim_latent':32,
    'dim_block': np.append([len(gene_names)], len(ADT_names)),
    'dist_block':['NB','NB'],
    'dim_block_enc':np.array([256, 128]),
    'dim_block_dec':np.array([256, 128]),
    'block_names':np.array(['rna', 'adt']),
    'uni_block_names':np.array(['rna','adt']),
    'dim_block_embed':np.array([32, 16]),

    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.15,0.85]),
    'beta_reverse':0.,

    "p_feat" : 0.2,
    "p_modal" : np.ones(2)/2,

}
config = SimpleNamespace(**config)


model = scVAEIT(config, data, None, batches)
del data
model.train(valid=False, num_epoch=300, batch_size=64, save_every_epoch=300,
        verbose=False, checkpoint_dir=path_root+'checkpoint/D'+dataset_id+'/')

latent =  model.get_latent_z()

np.savetxt(results_path+dataset_num+"_latent_scVAEIT.csv", latent, delimiter=',')
