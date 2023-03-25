#!/usr/bin/env python
# coding: utf-8

# In[52]:


import scvi
import numpy as np
import scanpy as sc
scvi.settings.seed = 420
from scipy.sparse import csr_matrix
from scipy.io import mmread
import pandas as pd
import anndata as ad


# In[53]:


def read_RNA_ATAC(RNA_path,ATAC_path):
    # gene expression
    cell_names = pd.read_csv(RNA_path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids'] 
    X = csr_matrix(mmread(RNA_path+'matrix.mtx').T)
    gene_names = pd.read_csv(RNA_path+'genes.tsv', sep = '\t',  header=None, index_col=None) 
    gene_names.columns =  ['gene_ids'] 
    adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
    adata_RNA.var_names_make_unique()
    # peak information
    cell_names = pd.read_csv(ATAC_path + 'barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids'] 
    X = csr_matrix(mmread(ATAC_path + 'matrix.mtx').T)
    peak_name = pd.read_csv(ATAC_path + 'peaks.bed',header=None,index_col=None)
    peak_name.columns = ['peak_ids']
    adata_ATAC  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
    return adata_RNA, adata_ATAC


# In[54]:


train_id = "Dataset35"
test_id = "Dataset36"
train_data_path = "../data/"+train_id
test_data_path = "../data/"+test_id
adata_paired_rna, adata_paired_atac = read_RNA_ATAC(train_data_path+"/RNA/",train_data_path+"/ATAC/")


# In[ ]:


adata_rna = read_RNA_ATAC(test_data_path+"/RNA/",None)


# In[56]:


adata = ad.concat([adata_RNA, adata_ATAC], merge = "same",axis=1)
adata.obs_keys
modality = ['Gene Expression']*adata_RNA.shape[1]+['Peaks']*adata_ATAC.shape[1]
adata.var['modality']=modality


# In[58]:


adata_rna.var['modality']=['Gene Expression']*adata_RNA.shape[1]


# In[60]:


adata_paired = ad.concat([adata_paired_rna, adata_paired_atac], merge = "same",axis=1)


# In[61]:


adata_paired.var['modality']=modality


# In[14]:


# We can now use the organizing method from scvi to concatenate these anndata
adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired, adata_rna)


# In[17]:


adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()
adata_mvi.var


# In[18]:


sc.pp.filter_genes(adata_mvi, min_cells=int(adata_mvi.shape[0] * 0.001))


# In[19]:


scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key='modality')


# In[20]:


mvi = scvi.model.MULTIVI(
    adata_mvi,
    n_genes=(adata_mvi.var['modality']=='Gene Expression').sum(),
    n_regions=(adata_mvi.var['modality']=='Peaks').sum(),
    n_proteins=0
)
mvi.view_anndata_setup()


# In[28]:


mvi.train(max_epochs=100)


# In[29]:


adata_mvi.obsm["MultiVI_latent"] = mvi.get_latent_representation()
sc.pp.neighbors(adata_mvi, use_rep="MultiVI_latent")
sc.tl.umap(adata_mvi, min_dist=0.2)
sc.pl.umap(adata_mvi, color='modality')


# In[30]:


imputed_expression = mvi.get_normalized_expression()


# In[31]:


imputed_accessibility = mvi.get_accessibility_estimates()


# In[32]:


imputed_accessibility.shape


# In[81]:


imputed_accessibility.to_csv('../results/MultiVI/MultiVI_'+ train_id + '_'+ test_id + '_pred.csv',index=False)


# In[65]:


imputed_accessibility.index=imputed_accessibility.iloc[:,0]


# In[66]:


imputed_accessibility=imputed_accessibility.drop(imputed_accessibility.columns[[0]], axis=1)


# In[67]:


imputed_accessibility.index.names=[None]


# In[68]:


# perd_acc = imputed_accessibility[adata_paired.n_obs:]
# obs_name = [name.split('_')[0] for name in list(perd_acc.index)]
perd_acc = imputed_accessibility[adata_paired.n_obs:]
obs_name = [name.rsplit('_',1)[0] for name in list(perd_acc.index)]


# In[46]:


perd_acc


# In[69]:


ref_acc = adata_ATAC[obs_name,list(imputed_accessibility)]


# In[70]:


ref_acc_df = pd.DataFrame(ref_acc.X.toarray(), columns= ref_acc.var_names,index= ref_acc.obs_names)


# In[71]:


ref_acc_df


# In[72]:


pcc_cell = [np.corrcoef(ref_acc_df.values[i,], perd_acc.values[i,])[0,1] for i in range(ref_acc_df.shape[0])] 
np.median(pcc_cell)


# In[74]:


pcc_peak = [np.corrcoef(ref_acc_df.values[:,i], perd_acc.values[:,i])[0,1] for i in range(ref_acc_df.shape[1])] 


# In[75]:


pcc_peak_df=pd.DataFrame(pcc_peak)
pcc_cell_df=pd.DataFrame(pcc_cell)


# In[76]:


np.nanmedian(pcc_peak)


# In[77]:


pcc_peak_df.to_csv('../results/MultiVI/MultiVI_'+ train_id + '_'+ test_id + '_pcc_peak.csv')
pcc_cell_df.to_csv('../results/MultiVI/MultiVI_'+ train_id + '_'+ test_id + '_pcc_cell.csv')


# In[ ]:





# In[ ]:




