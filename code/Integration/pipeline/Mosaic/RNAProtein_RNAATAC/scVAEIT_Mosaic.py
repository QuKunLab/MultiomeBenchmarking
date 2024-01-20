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
sys.path.append("../")
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

arguments = sys.argv

if not os.path.exists(arguments[1]):
    print("Input RNA+ATAC Data does not exist.")

if not os.path.exists(arguments[2]):
    print("Input RNA+Protein Data does not exist.")

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
adata_RNA.var_names_make_unique()
sc.pp.filter_cells(adata_RNA, min_genes=1)
sc.pp.filter_genes(adata_RNA, min_cells=20)

# peak information
cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
peak_name = pd.read_csv(path+'/ATAC/features.tsv',header=None,index_col=None)
peak_name.columns = ['peak_ids']
adata_ATAC  = sc.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
sc.pp.filter_cells(adata_ATAC, min_genes=1)
sc.pp.filter_genes(adata_ATAC, min_cells=1)

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

# Protein information
adata_adt = sc.read_csv(path+"/ADT.csv").T


batches = [0] * adata_RNA.shape[0] + [1] * adata_rna.shape[0]
batches = np.array(batches)
batches = batches.reshape((batches.shape[0],1))


overlap_gene = set(adata_rna.var_names).intersection(adata_RNA.var_names)
adata_RNA = adata_RNA[:,list(overlap_gene)]
adata_rna = adata_rna[:,adata_RNA.var_names]

id_peak_XY = ~(np.char.startswith(adata_ATAC.var_names.tolist(), 'chrX') | np.char.startswith(adata_ATAC.var_names.tolist(), 'chrY'))
id_peak_chr = np.char.startswith(adata_ATAC.var_names.tolist(), 'chr')
adata_ATAC = adata_ATAC[:,id_peak_XY&id_peak_chr]
sc.pp.highly_variable_genes(
    adata_ATAC,
    flavor="seurat_v3",
    n_top_genes=30000,
    subset=True)

# start_time = time.time()
adata = sc.concat([adata_RNA, adata_rna], axis=0)
adata.obs['batch'] = ['batch1']*adata_RNA.n_obs + ['batch2']*adata_rna.n_obs
ncell_RNA = adata_RNA.shape[0]
ncell_rna = adata_rna.shape[0]
del adata_rna,adata_RNA

sc.pp.highly_variable_genes(
    adata,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=False)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata = adata[:, adata.var.highly_variable].copy()
gene_names = adata.var_names
print(len(gene_name))
X = adata.X.toarray()
del adata

chunk_atac = np.array([
    np.sum(np.char.startswith(adata_ATAC.var_names.tolist(), 'chr%d-'%i)) for i in range(1,23) #people is 23 and mouse is 20
    ], dtype=np.int32)
dim_input_arr = np.array([len(gene_names),len(adata_adt.var_names), len(adata_ATAC.var_names)])
print(dim_input_arr)



sc.pp.normalize_total(adata_adt, target_sum=1e4)
sc.pp.log1p(adata_adt)
ADT_names = adata_adt.var_names
Y = np.vstack((np.zeros((ncell_RNA, adata_adt.shape[1])), adata_adt.X))
del adata_adt

Z = adata_ATAC.X
Z[Z>0.] = 1.
Z = Z.toarray()
Z = np.vstack((Z, np.zeros((ncell_rna, adata_ATAC.shape[1]))))
del adata_ATAC

data = np.c_[X, Y, Z]
del X, Y, Z

id_X_Batch1 = np.array(range(dim_input_arr[0]), dtype=np.int32)
id_Y_Batch1 = np.array(range(dim_input_arr[1]), dtype=np.int32)  
id_Z_Batch1 = np.array(range(dim_input_arr[2]), dtype=np.int32)  

masks = - np.ones((2, np.sum(dim_input_arr)), dtype=np.float32)
masks[0,id_X_Batch1] = 0.
masks[0,id_Z_Batch1 + np.sum(dim_input_arr[:2])] = 0.
masks[1,id_X_Batch1] = 0.
masks[1,id_Y_Batch1 + dim_input_arr[0]] = 0.
masks = tf.convert_to_tensor(masks, dtype=tf.float32)

path_root = './scVAEIT_res/'
config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256], 
    'dim_latent':32,
    'dim_block': np.append([len(gene_names),len(ADT_names)], chunk_atac), 
    'dist_block':['NB','NB'] + ['Bernoulli' for _ in chunk_atac], 
    'dim_block_enc':np.array([256, 128] + [16 for _ in chunk_atac]),
    'dim_block_dec':np.array([256, 128] + [16 for _ in chunk_atac]),
    'block_names':np.array(['rna', 'adt'] + ['atac' for _ in range(len(chunk_atac))]),
    'uni_block_names':np.array(['rna','adt','atac']),
    'dim_block_embed':np.array([32, 16] + [2 for _ in range(len(chunk_atac))]),

    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.15,0.83,0.02]),
    'beta_reverse':0.,

    "p_feat" : 0.2,
    "p_modal" : np.ones(3)/3,
    
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
