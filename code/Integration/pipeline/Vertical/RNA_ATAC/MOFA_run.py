import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import muon as mu
from scipy.io import mmread
from scipy.sparse import csr_matrix
import os, sys, time
from muon import MuData
import importlib
# sys.path.append(".")
# sys.path.append("./mofapy2/")
import mofapy2

arguments = sys.argv

if not os.path.exists(arguments[1]):
    print("Input 10xMultiome Data does not exist.")
  
if not os.path.exists(arguments[2]):
    print("Don't input DataName.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")
print("Data Name: "+ arguments[2] + " \n")

# 10x multiome
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

sc.pp.highly_variable_genes(
    adata_ATAC,
    flavor="seurat_v3",
    n_top_genes=30000,
    subset=False
)

sc.pp.highly_variable_genes(
    adata_RNA,
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=False
)
# adata_RNA.layers["counts"] = adata_RNA.X.copy()
sc.pp.normalize_total(adata_RNA, target_sum=1e4)
sc.pp.log1p(adata_RNA)
adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable].copy()
# rna.raw = rna
# adata_RNA.layers["lognorm"] = adata_RNA.X.copy()

# adata_ATAC.layers["counts"] = adata_ATAC.X.copy()
sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
sc.pp.log1p(adata_ATAC)
adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable].copy()
# adata_ATAC.layers["lognorm"] = adata_ATAC.X.copy()

mdata = MuData({'rna': adata_RNA, 'atac': adata_ATAC})
del adata_RNA, adata_ATAC

start_time = time.time() #From now on time

mu.tl.mofa(mdata, gpu_mode=False,outfile="./models/MOFA.hdf5",use_var=None)

# mdata.write("mudata_GPU.h5mu")
# mdata.obsm['X_mofa'].shape
# mdata_CPU = mu.read("./mudata_CUP.h5mu")
np.savetxt(arguments[2]+"MOFA.csv", mdata.obsm['X_mofa'], delimiter=',')
end_time = time.time()

execution_time = end_time - start_time
print(f"running time:{execution_time} s")


