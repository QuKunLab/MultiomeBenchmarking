import scarches as sca
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
import gdown
import json
import os
import sys

import warnings
warnings.filterwarnings("ignore")

arguments = sys.argv

if not os.path.exists(arguments[1]):
    print("Input Data does not exist.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")

data_path = arguments[1]

RNA_path = data_path+"/RNA/"
cell_names = pd.read_csv(RNA_path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
cell_names['cell_ids'] = cell_names['cell_ids'].str.replace('.','-')
X = csr_matrix(mmread(RNA_path+'matrix.mtx').T)
gene_names = pd.read_csv(RNA_path+'features.tsv', sep = '\t',  header=None, index_col=None) 
gene_names.columns =  ['gene_ids'] 
adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA.var_names_make_unique()
del X

adata_RNA.layers["counts"] = adata_RNA.X.copy()
sc.pp.normalize_total(adata_RNA)
sc.pp.log1p(adata_RNA)
sc.pp.highly_variable_genes(
    adata_RNA,
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=False
)
adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable].copy()

gene_list = pd.DataFrame(data = list(adata_RNA.var_names),columns = ['gene_ids'])
gene_list.to_csv(arguments[2]+"/DEgene.csv")