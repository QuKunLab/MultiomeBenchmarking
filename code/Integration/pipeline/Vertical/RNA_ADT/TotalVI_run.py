import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mudata as md
import muon
from scipy.io import mmread
from scipy.sparse import csr_matrix
import scanpy as sc
import scvi
import torch
import time
import os
import sys

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

# read ADT data
adata_ADT = pd.read_csv(data_path+'/ADT.csv',index_col = 0)
adata_ADT.columns = adata_ADT.columns.str.replace('.','-')
adata_ADT.index = adata_ADT.index.str.replace('.','_')
adata_ADT.index = adata_ADT.index.str.replace('-','_')
adata_ADT = ad.AnnData(adata_ADT.T)
modality = ['batch1']*adata_RNA.shape[0]
adata_RNA.obs['batch']=modality

start_time = time.time()

mdata = md.MuData({"rna": adata_RNA, "protein": adata_ADT})
# Place subsetted counts in a new modality
sc.pp.highly_variable_genes(
    mdata.mod["rna"],
    n_top_genes=4000,
    flavor="seurat_v3",
    batch_key="batch",
    layer="counts",
)
mdata.mod["rna_subset"] = mdata.mod["rna"][
    :, mdata.mod["rna"].var["highly_variable"]].copy()
mdata.update()

scvi.model.TOTALVI.setup_mudata(
    mdata,
    rna_layer="counts",
    protein_layer=None,
    batch_key="batch",
    modalities={
        "rna_layer": "rna_subset",
        "protein_layer": "protein",
        "batch_key": "rna_subset",
    },
)

# run TotalVI
model = scvi.model.TOTALVI(mdata)
model.train()


adata_RNA.obsm["X_totalVI"] = model.get_latent_representation()
latent = adata_RNA.obsm['X_totalVI'].copy()
np.savetxt(arguments[2]+"/TotalVI.csv", latent, delimiter=',')

