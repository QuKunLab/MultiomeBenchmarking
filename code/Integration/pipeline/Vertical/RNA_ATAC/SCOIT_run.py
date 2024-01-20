import numpy as np
import pandas as pd
from scoit import sc_multi_omics
import time
import os
import sys
import scanpy as sc
import scipy
from scoit import sc_multi_omics, unionCom_alignment
from sklearn.decomposition import TruncatedSVD

arguments = sys.argv

if not os.path.exists(arguments[1]):
    print("Input Data does not exist.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")

def TF_IDF(peak_raw):
      row_sum = peak_raw.sum(axis=1)
      col_sum = peak_raw.sum(axis=0)
      peak_raw_s = scipy.sparse.csr_matrix(peak_raw)
      mat_row = scipy.sparse.dia_matrix((1/(row_sum.T+1e-8), 0), shape=(row_sum.shape[0], row_sum.shape[0]))
      mat_col = scipy.sparse.dia_matrix((1/(col_sum+1e-8), 0), shape=(col_sum.shape[1], col_sum.shape[1]))
      return (mat_row.dot(peak_raw_s)).dot(mat_col)

path = arguments[1]

cellinfo = pd.read_csv(path + '/RNA/barcodes.tsv',header=None)
cellinfo.columns = ['cell_id']
geneinfo = pd.read_csv(path + '/RNA/features.tsv',header=None)
geneinfo.columns = ['gene_id']
adata = sc.read(path + '/RNA/matrix.mtx',cache=True).T
adata_rna = sc.AnnData(adata.X, obs=cellinfo ,var=geneinfo)
adata_rna.var_names = adata_rna.var['gene_id']
adata_rna.obs_names = adata_rna.obs['cell_id']
adata_rna.obs["data_type"] = "rna"

# CMP
sc.pp.normalize_total(adata_rna, target_sum=1e6)

# log transformation
sc.pp.log1p(adata_rna)
sc.pp.highly_variable_genes(adata_rna)
adata_rna = adata_rna[:, adata_rna.var.highly_variable]

sc.pp.scale(adata_rna)
sc.tl.pca(adata_rna, svd_solver='arpack')

expression_data = adata_rna.obsm['X_pca']

cellinfo = pd.read_csv(path + '/ATAC/barcodes.tsv',header=None)
cellinfo.columns = ['cell_id']
geneinfo = pd.read_csv(path + '/ATAC/features.tsv',header=None)
geneinfo.columns = ['gene_id']
adata = sc.read(path + '/ATAC/matrix.mtx',cache=True).T

adata_atac = sc.AnnData(adata.X, obs=cellinfo ,var=geneinfo)
adata_atac.var_names = adata_atac.var['gene_id']
adata_atac.obs_names = adata_atac.obs['cell_id']
adata_atac.obs["data_type"] = "atac"

# TF-IDF
adata_atac.X = TF_IDF(adata_atac.X)

# LSI
svd = TruncatedSVD(n_components=50, algorithm='arpack',random_state=123)
svd.fit(adata_atac.X.T)
U_ATAC = np.diag(svd.singular_values_).dot(svd.components_)
ATAC_data= np.array(U_ATAC).T

start_time = time.time()
data = [expression_data, ATAC_data]
print(data[0].shape)
print(data[1].shape)

sc_model = sc_multi_omics()
predict_data = sc_model.fit_list_complete(data, normalization=False, dist="gaussian", lr=1e-3, n_epochs=3000)
np.savetxt(arguments[2]+"/SCOIT.csv", sc_model.C, delimiter = ',')

print(time.time() - start_time)

