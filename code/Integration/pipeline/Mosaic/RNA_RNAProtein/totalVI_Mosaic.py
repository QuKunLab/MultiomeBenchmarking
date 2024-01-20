from scipy.io import mmread
from scipy.sparse import csr_matrix
import anndata as ad
import pandas as pd
import scvi
import numpy as np
import scanpy as sc
import os, time, sys

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

arguments = sys.argv
if not os.path.exists(arguments[1]):
    print("Input RNA+Protein Data does not exist.")
if not os.path.exists(arguments[2]):
    print("Input RNA Data does not exist.")
if not os.path.exists(arguments[3]):
    print("Don't input DataName.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")
print("Input file: "+ arguments[2] + " \n")
print("Data Name: "+ arguments[3] + " \n")

## CITE-seq
path = arguments[1]
# gene expression
cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_names = pd.read_csv(path+'/RNA/features.tsv', sep = '\t',  header=None, index_col=None) 
gene_names.columns =  ['gene_ids'] 
adata_CITE = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_CITE.var_names_make_unique()
sc.pp.filter_cells(adata_CITE, min_genes=1)
sc.pp.filter_genes(adata_CITE, min_cells=20)

adata_ADT = sc.read_csv(path+"/ADT.csv").T
sc.pp.filter_cells(adata_ADT, min_genes=1)
sc.pp.filter_genes(adata_ADT, min_cells=1)
adata_CITE.obsm['protein_expression'] = adata_ADT.X
protein_name = adata_ADT.var_names
del X, adata_ADT

path = arguments[2]
## scRNA-seq
cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids'] 
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_name = pd.read_csv(path+'/RNA/features.tsv',header=None,index_col=None)
gene_name.columns = ['gene_ids']
adata_rna  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_name.gene_ids))
sc.pp.filter_cells(adata_rna, min_genes=1)
sc.pp.filter_genes(adata_rna, min_cells=20)

del X
adata_rna.obsm['protein_expression'] =np.zeros((adata_rna.shape[0],len(protein_name)))
adata = ad.concat([adata_CITE, adata_rna],axis=0)
adata.obs_names_make_unique()

adata.obs['batch'] = ['batch1']*adata_CITE.shape[0]+['batch2']*adata_rna.shape[0]
del adata_rna,adata_CITE

sc.pp.highly_variable_genes(
    adata, batch_key="batch", flavor="seurat_v3", n_top_genes=4000, subset=True
)

scvi.settings.seed = 12345
scvi.model.TOTALVI.setup_anndata( adata, batch_key="batch", protein_expression_obsm_key="protein_expression" )
model = scvi.model.TOTALVI(adata, latent_distribution="normal", n_layers_decoder=2)
model.train()

# adata.obsm["X_totalVI"] = model.get_latent_representation()
latent = model.get_latent_representation()
np.savetxt(arguments[3]+"/totalVI.csv", latent, delimiter=',')
# end_time = time.time()

# execution_time = end_time - start_time
# print(f"running time:{execution_time} s")
