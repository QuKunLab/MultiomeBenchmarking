import os
import sys
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
sys.path.append('../../')
from scVAEIT.VAEIT import scVAEIT
import time

import warnings
warnings.filterwarnings("ignore")

arguments = sys.argv

if not os.path.exists(arguments[1]):
    print("Input Data does not exist.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")

path = arguments[1]

cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_names = pd.read_csv(path+'/RNA/features.tsv', sep = '\t',  header=None, index_col=None)
gene_names.columns =  ['gene_ids']
adata_RNA = sc.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
del X
adata_RNA.var_names_make_unique()
# sc.pp.filter_cells(adata_RNA, min_genes=1)
sc.pp.filter_genes(adata_RNA, min_cells=20)

# peak information
cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
peak_name = pd.read_csv(path+'/ATAC/features.tsv',header=None,index_col=None)
peak_name.columns = ['peak_ids']
adata_ATAC  = sc.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
del X
# sc.pp.filter_cells(adata_ATAC, min_genes=1)
sc.pp.filter_genes(adata_ATAC, min_cells=1)

batches = [0] * adata_RNA.shape[0]
batches = np.array(batches)
batches = batches.reshape((batches.shape[0],1))

id_peak_XY = ~(np.char.startswith(adata_ATAC.var_names.tolist(), 'chrX') | np.char.startswith(adata_ATAC.var_names.tolist(), 'chrY'))
id_peak_chr = np.char.startswith(adata_ATAC.var_names.tolist(), 'chr')
adata_ATAC = adata_ATAC[:,id_peak_XY&id_peak_chr]

chunk_atac = np.array([
    np.sum(np.char.startswith(adata_ATAC.var_names.tolist(), 'chr%d-'%i)) for i in range(1,23) #people is 23 and mouse is 20
    ], dtype=np.int32)
dim_input_arr = np.array([len(adata_RNA.var_names),len(adata_ATAC.var_names)])
print(dim_input_arr)
print(chunk_atac)

start_time = time.time()
Z = adata_ATAC.X
Z[Z>0.] = 1.
Z = Z.toarray()

sc.pp.normalize_total(adata_RNA, target_sum=1e4)
sc.pp.log1p(adata_RNA)

X = adata_RNA.X.toarray()
del adata_RNA, adata_ATAC

data = np.c_[X, Z]
del X, Z

id_X_Batch1 = np.array(range(dim_input_arr[0]), dtype=np.int32)
id_Z_Batch1 = np.array(range(dim_input_arr[1]), dtype=np.int32)


masks = np.zeros((2, np.sum(dim_input_arr)), dtype=np.float32)

masks = tf.convert_to_tensor(masks, dtype=tf.float32)
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


model = scVAEIT(config, data, masks, batches)

del data, masks
model.train(valid=False, num_epoch=300, batch_size=64, save_every_epoch=300,
        verbose=False, checkpoint_dir='./checkpoint/')

latent =  model.get_latent_z()
np.savetxt(arguments[2]+"/scVAEIT.csv", model.get_latent_z(), delimiter=',')