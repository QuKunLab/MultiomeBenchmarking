import os
import sys
sys.path.append("../pipeline/scVAEIT/")
import anndata as ad
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
sys.path.append("../")
from scVAEIT.VAEIT import scVAEIT
import time

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
    locals()['adata_RNA'+str(i)] = sc.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
    del X
    eval('adata_RNA'+str(i)).var_names_make_unique()
    sc.pp.filter_cells(eval('adata_RNA'+str(i)), min_genes=1)
    sc.pp.filter_genes(eval('adata_RNA'+str(i)), min_cells=20)

    # peak information
    cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids']
    X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
    peak_name = pd.read_csv(path+'/ATAC/peaks.tsv',header=None,index_col=None)
    peak_name.columns = ['peak_ids']
    locals()['adata_ATAC'+str(i)] = sc.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
    del X
    sc.pp.filter_cells(eval('adata_ATAC'+str(i)), min_genes=1)
    sc.pp.filter_genes(eval('adata_ATAC'+str(i)), min_cells=1)

batches = [0] * adata_RNA1.shape[0] + [1] * adata_RNA2.shape[0]
batches = np.array(batches)
batches = batches.reshape((batches.shape[0],1))

adata = ad.concat([adata_RNA1, adata_RNA2],axis=0)
adata.obs['batch'] = ['batch1']*adata_RNA1.n_obs + ['batch2']*adata_RNA2.n_obs
sc.pp.highly_variable_genes(
    adata,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=False )
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata = adata[:, adata.var.highly_variable].copy()
X = adata.X.toarray()

for i in range(1,3):
    id_peak_XY = ~(np.char.startswith(eval('adata_ATAC'+str(i)).var_names.tolist(), 'chrX') | np.char.startswith(eval('adata_ATAC'+str(i)).var_names.tolist(), 'chrY'))
    id_peak_chr = np.char.startswith(eval('adata_ATAC'+str(i)).var_names.tolist(), 'chr')
    locals()['adata_ATAC'+str(i)] = eval('adata_ATAC'+str(i))[:,id_peak_XY&id_peak_chr]

# overlap_peak = set(adata_ATAC1.var_names).intersection(adata_ATAC2.var_names)
# adata_ATAC1 = adata_ATAC1[:,list(overlap_peak)]
# adata_ATAC2 = adata_ATAC2[:,list(overlap_peak)]

adata_ATAC = ad.concat([adata_ATAC1, adata_ATAC2],axis=0)
adata_ATAC.obs['batch'] = ['batch1']*adata_ATAC1.n_obs + ['batch2']*adata_ATAC2.n_obs
sc.pp.highly_variable_genes(
    adata_ATAC,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=30000,
    subset=False )

chunk_atac = np.array([
    np.sum(np.char.startswith(adata_ATAC.var_names.tolist(), 'chr%d-'%i)) for i in range(1,23) #people is 23 and mouse is 20
    ], dtype=np.int32)

gene_names = adata.var_names
dim_input_arr = np.array([4000,adata_ATAC.shape[1]])
print(dim_input_arr)
print(chunk_atac)

Z = adata_ATAC.X
Z[Z>0.] = 1.
Z = Z.toarray()

del adata_RNA1, adata_RNA2, adata_ATAC1, adata_ATAC2, adata, adata_ATAC

data = np.c_[X, Z]
del X, Z

id_X_Batch1 = np.array(range(dim_input_arr[0]), dtype=np.int32)
id_Z_Batch1 = np.array(range(dim_input_arr[1]), dtype=np.int32)
id_X_Batch2 = np.array(range(dim_input_arr[0]), dtype=np.int32)
id_Z_Batch2 = np.array(range(dim_input_arr[1]), dtype=np.int32)

#masks = np.zeros((2, np.sum(dim_input_arr)), dtype=np.float32)
# masks[0,id_X_Batch1] = 0.
# masks[0,id_Z_Batch1+dim_input_arr[0]] = 0.
# masks[1,id_X_Batch2] = 0.

#masks = tf.convert_to_tensor(masks, dtype=tf.float32)
path_root = './scVAEIT_res/'

config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256],
    'dim_latent':32,
    'dim_block': np.append([len(gene_names)], chunk_atac),
    'dist_block':['NB'] + ['Bernoulli' for _ in chunk_atac],
    'dim_block_enc':np.array([256] + [16 for _ in chunk_atac]),
    'dim_block_dec':np.array([256] + [16 for _ in chunk_atac]),
    'block_names':np.array(['rna'] + ['atac' for _ in range(len(chunk_atac))]),
    'uni_block_names':np.array(['rna','atac']),
    'dim_block_embed':np.array([32] + [2 for _ in range(len(chunk_atac))]),

    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.88,0.12]),
    'beta_reverse':0.,

    "p_feat" : 0.2,
    "p_modal" : np.ones(2)/2,
}
config = SimpleNamespace(**config)


model = scVAEIT(config, data, None, batches)
del data
model.train(valid=False, num_epoch=300, batch_size=64, save_every_epoch=300,
        verbose=False, checkpoint_dir=path_root+'checkpoint/'+dataset_id+'/')

latent =  model.get_latent_z()
np.savetxt(save_path+dataset_id+"_latent_scVAEIT.csv", latent, delimiter=',')
