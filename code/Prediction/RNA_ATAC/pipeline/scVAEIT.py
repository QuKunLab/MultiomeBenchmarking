import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
os.environ["OMP_NUM_THREADS"] = "11"
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "11" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "11" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_CACHE_DIR"]='/tmp/numba_cache'
os.environ["TF_GPU_ALLOCATOR"]='cuda_malloc_async'

import numpy as np
import pandas as pd
import scipy as sp
import scanpy as sc
import scipy.sparse
import h5py
import tensorflow as tf
import tensorflow_probability as tfp

physical_devices = tf.config.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
    
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar
tfd = tfp.distributions
import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.io import mmread, mmwrite
X = csr_matrix(mmread(f'./Dataset37/Train/RNA/matrix.mtx').T).toarray()
gene_names = pd.read_csv(f'./Dataset37/Train/RNA/features.tsv', sep = '\t',  header=None, index_col=None).values.flatten()
Z = csr_matrix(mmread(f'./Dataset37/Train/ATAC/matrix.mtx').T).toarray()
peak_names = pd.read_csv(f'./Dataset37/Train/ATAC/features.tsv', sep = '\t',  header=None, index_col=None).values.flatten()


peak_names = np.array(peak_names).astype(str)
gene_names = np.array(gene_names).astype(str)
id_peak_XY =np.char.startswith(peak_names, 'chr')&(~(np.char.startswith(peak_names, 'chrX') | np.char.startswith(peak_names, 'chrY')))
peak_names = peak_names[id_peak_XY]
Z_fil = Z[:,id_peak_XY]
del Z
len(peak_names)
chunk_atac = np.array([
    np.sum(np.char.startswith(peak_names, 'chr%d-'%i)) for i in range(1,23) #6是染色体的数目
    ], dtype=np.int32)
### RNA-> ATAC
dim_input_arr = [X.shape[1], Z_fil.shape[1]]
config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256], 
    'dim_latent':32,
    'dim_block': np.append([len(gene_names)], chunk_atac), 
    'dist_block':['NB'] + ['Bernoulli' for _ in chunk_atac], 
    'dim_block_enc':np.array([256] + [16 for _ in chunk_atac]),
    'dim_block_dec':np.array([256] + [16 for _ in chunk_atac]),
    'dim_block_embed':np.array([32] + [2 for _ in chunk_atac]),
    
    'block_names':np.array(['rna'] + ['atac' for _ in range(len(chunk_atac))]),
    'uni_block_names':np.array(['rna','atac']),
    
    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.94,0.06]),
    'beta_reverse':0.,

    "p_feat" : 0.2,
    "p_modal" : np.ones(2)/2,
    
}
config = SimpleNamespace(**config)
n_samples = 50
Z_fil = (Z_fil>0).astype(np.float32)
X = np.log(X/np.sum(X, axis=1, keepdims=True)*1e4+1.)
data = np.c_[X, Z_fil].astype(np.float32)
data.shape
from scVAEIT.VAEIT import scVAEIT
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
masks = np.zeros((1,data.shape[1]), dtype=np.float32)
model = scVAEIT(config, data, masks)
import time 

start_time = time.time()
path_root = './scVAEIT-main/result/ex2/pbmc_unsorted_3k/'
model.train(
    valid=False, num_epoch=300, batch_size=512,  save_every_epoch=100, # 300 50 
    verbose=True, checkpoint_dir=path_root+'checkpoint/'
)
use_time = (time.time() - start_time)
print(use_time)
X = csr_matrix(mmread(f'./Dataset37/Test/RNA/matrix.mtx').T).toarray()
gene_names = pd.read_csv(f'./Dataset37/Test/RNA/features.tsv', sep = '\t',  header=None, index_col=None).values.flatten()
Z = csr_matrix(mmread(f'./Dataset37/Test/ATAC/matrix.mtx').T).toarray()
peak_names = pd.read_csv(f'./Dataset37/Test/ATAC/features.tsv', sep = '\t',  header=None, index_col=None).values.flatten()
cell_names = pd.read_csv(f'./Dataset37/Test/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None).values.flatten()
cell_names = pd.read_csv(f'./Dataset37/Test/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None).values.flatten()
cell_names = np.array(cell_names).astype(str)
X = np.log(X/np.sum(X, axis=1, keepdims=True)*1e4+1.)

peak_names = np.array(peak_names).astype(str)
gene_names = np.array(gene_names).astype(str)
id_peak_XY =np.char.startswith(peak_names, 'chr')&(~(np.char.startswith(peak_names, 'chrX') | np.char.startswith(peak_names, 'chrY')))
peak_names = peak_names[id_peak_XY]
Z_fil = Z[:,id_peak_XY]
del Z

# data spliting
test = np.c_[X, np.zeros(Z_fil.shape)].astype(np.float32)
del X
mask_atac = np.zeros((1,np.sum(model.vae.config.dim_input_arr)), dtype=np.float32)
dataset_test = tf.data.Dataset.from_tensor_slices((
    test,
    model.cat_enc.transform(np.zeros((test.shape[0],1))).toarray().astype(np.float32),
    np.zeros(test.shape[0]).astype(np.int32)
)).batch(512).prefetch(tf.data.experimental.AUTOTUNE)
mask_atac[:,-model.vae.config.dim_input_arr[-1]:] = -1.
recon = model.vae.get_recon(dataset_test, mask_atac)
Z_hat = recon[:, -model.vae.config.dim_input_arr[-1]:]
Z_hat.shape
Z_fil.shape
save_path = './Results/'
pred = pd.DataFrame(data=Z_hat,index=cell_names,columns=peak_names)
true = pd.DataFrame(data=Z_fil,index=cell_names,columns=peak_names)
pred.to_hdf(save_path + 'scVAEIT_pred.h5', 'a')
true.to_hdf(save_path + 'scVAEIT_true.h5', 'a')

    
    