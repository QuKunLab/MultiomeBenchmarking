import numpy as np
import anndata as ad
import pandas as pd
from scoit import sc_multi_omics
import time
import os
import sys
import scanpy as sc
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix,lil_matrix
from scipy.io import mmread, mmwrite
import scipy
sys.path.append("/home/math/hyl2016/Intergration_Benchmark/testcode_20230710/SCOIT-main")
from scoit import sc_multi_omics, unionCom_alignment
from sklearn.decomposition import TruncatedSVD

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

# filter genes
sc.pp.filter_genes(adata_rna, min_cells=10)
# CMP
sc.pp.normalize_total(adata_rna, target_sum=1e6)

# log transformation
sc.pp.log1p(adata_rna)
sc.pp.highly_variable_genes(
    adata_rna,
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=False
)
adata_rna = adata_rna[:,adata_rna.var['highly_variable']]

min_values = adata_rna.X.min(axis=0).toarray()
max_values = adata_rna.X.max(axis=0).toarray()
expression_data = (adata_rna.X - min_values) / (max_values - min_values)

# Protein clr
def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""
    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else adata.X)
    )
    return adata

adata_adt = sc.read_csv(path + '/ADT.csv').T
# filter genes
sc.pp.filter_genes(adata_adt, min_cells=10)
# CLR
adata_adt = clr_normalize_each_cell(adata_adt)
# log transformation
sc.pp.log1p(adata_adt)

min_values = np.min(adata_adt.X, axis=0)
max_values = np.max(adata_adt.X, axis=0)
adata_adt.X = (adata_adt.X - min_values) / (max_values - min_values)

data = [adata_rna.X.toarray(), adata_adt.X]
print(data[0].shape)
print(data[1].shape)

sc_model = sc_multi_omics()
predict_data = sc_model.fit_list_complete(data, dist="gaussian", lr=1e-3, n_epochs=5000)

np.savetxt(arguments[2]+"/SCOIT.csv", sc_model.C, delimiter = ',')
