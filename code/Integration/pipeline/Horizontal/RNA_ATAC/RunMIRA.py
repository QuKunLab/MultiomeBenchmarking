from scipy.io import mmread
from scipy.sparse import csr_matrix
import anndata as ad
import os, sys, time
import mira
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.getLogger().setLevel(logging.INFO)

dataset_id = "Dataset1"
save_path = "../results/Horizontal/RNA_ATAC/"  ## path to save results
data_path = "../dataset/Horizontal/RNA_ATAC/"+dataset_id ## path to raw data

for i in range(1,3):
    batch_num = str(i)
    path = data_path+"batch"+batch_num
    cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids']
    X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
    gene_names = pd.read_csv(path+'/RNA/genes.tsv', sep = '\t',  header=None, index_col=None)
    gene_names.columns =  ['gene_ids']
    adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
    adata_RNA.var_names_make_unique()
    # peak information
    cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids']
    X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
    peak_name = pd.read_csv(path+'/ATAC/peaks.tsv',header=None,index_col=None)
    peak_name.columns = ['peak_ids']
    adata_ATAC = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))

    locals()['adata_paired'+str(i)] = ad.concat([adata_RNA, adata_ATAC], merge = "same",axis=1)
    eval('adata_paired'+str(i)).obs_keys
    modality = ['Gene Expression']*adata_RNA.shape[1]+['Peaks']*adata_ATAC.shape[1]
    locals()['adata_paired'+str(i)].var['modality']=modality
    del adata_RNA, adata_ATAC

adata = ad.concat([adata_paired1, adata_paired2],axis=0)
adata.obs['batch'] = ['batch1']*adata_paired1.n_obs + ['batch2']*adata_paired2.n_obs
modality = ['Gene Expression']*adata.shape[1]
for i in np.where(adata.var_names.str.contains("chr"))[0]:
    modality[i] = 'Peaks'
adata.var['modality']=modality
del adata_paired1, adata_paired2

adata_rna = adata[:,adata.var['modality']=='Gene Expression']
adata_atac = adata[:,adata.var['modality']=='Peaks']
del adata

adata_rna.var.index = adata_rna.var.index.str.upper()
adata_rna.var_names_make_unique()

sc.pp.filter_genes(adata_rna, min_cells=20)
rawdata = adata_rna.X.copy()

sc.pp.highly_variable_genes(
    adata_rna,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=False
)

sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)
# sc.pp.highly_variable_genes(adata_rna, min_disp = 0.0001,n_top_genes=2990)
adata_rna.layers['counts'] = rawdata
adata_rna = adata_rna[:, adata_rna.var.highly_variable].copy()
del rawdata

model_rna = mira.topics.make_model(
    adata_rna.n_obs, adata_rna.n_vars, # helps MIRA choose reasonable values for some hyperparameters which are not tuned.
    feature_type = 'expression',
    highly_variable_key='highly_variable',
    counts_layer='counts',
    categorical_covariates='batch'
)

learn_rate = model_rna.get_learning_rate_bounds(adata_rna)
model_rna.set_learning_rates(learn_rate[0], learn_rate[1]) # for larger datasets, the default of 1e-3, 0.1 usually works well.
# model_rna.plot_learning_rate_bounds(figsize=(7,3))
topic_contributions = mira.topics.gradient_tune(model_rna, adata_rna)
NUM_TOPICS = int(sum(np.array(topic_contributions)>0.05))
# mira.pl.plot_topic_contributions(topic_contributions, NUM_TOPICS)
#NUM_TOPICS = 15
model_rna = model_rna.set_params(num_topics = NUM_TOPICS).fit(adata_rna)
model_rna.predict(adata_rna)


sc.pp.filter_cells(adata_atac, min_genes=1)
sc.pp.filter_genes(adata_atac, min_cells=1)
### Optional, alternatively, mark informative or highly variable peaks here
np.random.seed(0)
adata_atac.var['endogenous_peaks'] = np.random.rand(adata_atac.shape[1]) <= min(1e7/adata_atac.shape[1], 1)

#sc.pp.highly_variable_genes(
#    adata_atac,
#    batch_key="batch",
#    flavor="seurat_v3",
#    n_top_genes=30000,
#    subset=True
#)

model_atac = mira.topics.make_model(
    *adata_atac.shape,
    feature_type = 'accessibility',
    endogenous_key='endogenous_peaks', # which peaks are used by the encoder network
    categorical_covariates='batch'
)

learn_rate = model_atac.get_learning_rate_bounds(adata_atac)
model_atac.set_learning_rates(learn_rate[0], learn_rate[1]) # for larger datasets, the default of 1e-3, 0.1 usually works well.
# model.plot_learning_rate_bounds(figsize=(7,3))
topic_contributions = mira.topics.gradient_tune(model_atac, adata_atac)

NUM_TOPICS = int(sum(np.array(topic_contributions)>0.05))
# mira.pl.plot_topic_contributions(topic_contributions, NUM_TOPICS)
model_atac = model_atac.set_params(num_topics = NUM_TOPICS).fit(adata_atac)

model_atac.predict(adata_atac)
adata_atac.obsm['X_umap_features'].shape

adata_rna, adata_atac = mira.utils.make_joint_representation(adata_rna, adata_atac)
latent = adata_atac.obsm['X_joint_umap_features']
np.savetxt(save_path+dataset_id+"_latent_MIRA.csv", latent, delimiter=',')
