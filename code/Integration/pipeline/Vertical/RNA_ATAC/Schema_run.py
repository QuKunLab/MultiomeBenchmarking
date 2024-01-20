import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import numpy as np
import sys
import scanpy as sc
import anndata as ad
import logging
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.io import mmread, mmwrite
import os 
import scipy.io as sio
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import scanpy as sc

import sys
sys.path.append('/home/math/hyl2016/Intergration_Benchmark/testcode_20230710/schema-master')
import schema
from schema import SchemaQP

arguments = sys.argv

if not os.path.exists(arguments[1]):
    print("Input 10xMultiome Data does not exist.")

if not os.path.exists(arguments[2]):
    print("Don't input DataName.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")
print("Data Name: "+ arguments[2] + " \n")

## 10x multiome
path = arguments[1]

cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_names = pd.read_csv(path+'/RNA/features.tsv', sep = '\t',  header=None, index_col=None)
gene_names.columns =  ['gene_ids']
adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA.var_names_make_unique()
sc.pp.filter_cells(adata_RNA, min_genes=10)
sc.pp.filter_genes(adata_RNA, min_cells=20)

# peak information
cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
peak_name = pd.read_csv(path+'/ATAC/features.tsv',header=None,index_col=None)
peak_name.columns = ['peak_ids']
adata_ATAC  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
sc.pp.filter_cells(adata_ATAC, min_genes=10)
sc.pp.filter_genes(adata_ATAC, min_cells=1)

adata_RNA.obs['data_type'] = 'RNA'
adata_ATAC.obs['data_type'] = 'ATAC'

import sklearn
svd2 = sklearn.decomposition.TruncatedSVD(n_components= 50, random_state = 17)
H2 = svd2.fit_transform(adata_ATAC.X)

sqp99 = schema.SchemaQP(min_desired_corr=0.99, mode='affine', params= {"decomposition_model":"nmf",
                                                      "num_top_components":50,
                                                      "do_whiten": 0,
                                                      "dist_npairs": 5000000})
dz99 = sqp99.fit_transform(adata_RNA.X, [H2], ['feature_vector'], [1])

np.savetxt(arguments[2]+"Schema.csv", dz99, delimiter=',')
