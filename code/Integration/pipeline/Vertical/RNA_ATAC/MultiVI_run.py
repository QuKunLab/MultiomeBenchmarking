from scipy.io import mmread
from scipy.sparse import csr_matrix
import anndata as ad
import pandas as pd
import scvi
import numpy as np
import scanpy as sc
scvi.settings.seed = 420
import os, sys, time
 
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
# gene expression
cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_names = pd.read_csv(path+'/RNA/features.tsv', sep = '\t',  header=None, index_col=None) 
gene_names.columns =  ['gene_ids'] 
adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA.var_names_make_unique()
sc.pp.filter_cells(adata_RNA, min_genes=10)
sc.pp.filter_genes(adata_RNA, min_cells=20)

sc.pp.highly_variable_genes(
    adata_RNA,
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=True
)
# peak information
cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
peak_name = pd.read_csv(path+'/ATAC/features.tsv',header=None,index_col=None)
peak_name.columns = ['peak_ids']
adata_ATAC  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
sc.pp.filter_cells(adata_ATAC, min_genes=10)
sc.pp.filter_genes(adata_ATAC, min_cells=1)

adata_paired = ad.concat([adata_RNA, adata_ATAC], merge = "same",axis=1) 
adata_paired.var['modality']=['Gene Expression']*adata_RNA.shape[1]+['Peaks']*adata_ATAC.shape[1]


# We can now use the organizing method from scvi to concatenate these anndata
adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired)
del adata_paired
adata_mvi.obs
adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()
adata_mvi.var
scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")

model = scvi.model.MULTIVI(
    adata_mvi,
    n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
    n_regions=(adata_mvi.var["modality"] == "Peaks").sum(),
)
model.view_anndata_setup()

model.train()

latent = model.get_latent_representation()

np.savetxt(arguments[2]+"MultiVI.csv", latent, delimiter=',')



