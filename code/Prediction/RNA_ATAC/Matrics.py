import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.io import mmread, mmwrite
import os
import numpy as np
import torch
import pandas as pd
from scipy.stats import spearmanr
import feather
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, precision_recall_curve
import pandas as pd
from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import PrecisionRecallDisplay

from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import anndata as ad
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, precision_recall_curve
import pandas as pd
from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import PrecisionRecallDisplay
def cal_auroc_all(pred,true):
    true[true > 1] = 1
    true_array = np.array(true,dtype=np.float32).flatten()
    pred_array = np.array(pred,dtype=np.float32).flatten()
    auroc = roc_auc_score(true_array,pred_array)
    aurocall_df = pd.DataFrame([auroc],columns=["auroc"])
    return aurocall_df

from sklearn import metrics
def cal_rmse_all(pred,true):

    true_array = np.array(true,dtype=np.float32).flatten()
    pred_array = np.array(pred,dtype=np.float32).flatten()
    
    rmse = metrics.mean_squared_error(true_array, pred_array)**0.5
    rmse_df = pd.DataFrame([rmse],columns=["rmse"])

    return rmse_df

import pandas as pd

def cal_cmd(pred, true):
    x = np.trace(pred.dot(true))
    y = np.linalg.norm(pred,'fro')*np.linalg.norm(true,'fro')
    cmd = 1- x/(y+1e-8)
    cmd_df = pd.DataFrame([cmd],columns=["cmd"])
    return cmd_df

import sys
arguments = sys.argv
method = arguments[1]
path = './Results/'
save_path = './Results/'
if method in ['Seurat','LIGER']:
    true = pd.read_feather(path +   f'{method}' + '_true.feather') 
    pred = pd.read_feather(path +   f'{method}' + '_pred.feather')
    pred.index = pred.loc[:,'index']
    pred = pred.drop('index',axis=1)
    true.index = true.loc[:,'index']
    true = true.drop('index',axis=1)
    true = true.T
    pred = pred.T
elif method in ['scMOG']:
    def change_peaknames(x):
        ing = x.split(':',1)
        return ing[0] + '-' + ing[1]
    # save_path = '/public/home/hanyu/Benchmark/results/scMOG/data/'
    pred_ad = ad.read_h5ad(f'{path}{method}_rna_atac_adata_final.h5ad')
    pred_ad.var_names = [change_peaknames(x) for x in pred_ad.var_names.tolist()]
    pred = pd.DataFrame(data=pred_ad.X.todense(),index=pred_ad.obs.values.flatten(),columns=pred_ad.var.index)
    del pred_ad
    path = f"./Dataset37/Test/"
    cell_names = pd.read_csv(path+'/ATAC/barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns = ['cell_ids'] 
    X = csr_matrix(mmread(path+'/ATAC/matrix.mtx').T)
    gene_names = pd.read_csv(path+'/ATAC/features.tsv', sep = '\t', header=None, index_col=None) 
    gene_names.columns = ['gene_ids'] 
    true_ad = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
    del X
    true = pd.DataFrame(data=true_ad.X.todense(),index=true_ad.obs.index,columns=true_ad.var.index)
    del true_ad
    overlap_peaks = list(set(true.columns).intersection(set(pred.columns)))
    pred = pred.loc[:,overlap_peaks]
    true = true.loc[:,overlap_peaks]
else:
    true = pd.read_hdf(path +   f'{method}'  + '_true.h5') 
    pred = pd.read_hdf(path +   f'{method}'  + '_pred.h5')
## PCC 
    
pcc_cell = [np.corrcoef(true.values[i,], pred.values[i,])[0,1] for i in range(true.shape[0])] 
pcc_peak = [np.corrcoef(true.values[:,i], pred.values[:,i])[0,1] for i in range(true.shape[1])] 
pcc_peak_df=pd.DataFrame(pcc_peak)
pcc_cell_df=pd.DataFrame(pcc_cell)

## CMD 

pred_array = pred.values
true_array = true.values
zero_rows_indices1 = list(np.where(~pred_array.any(axis=1))[0])
zero_rows_indices2 = list(np.where(~true_array.any(axis=1))[0])
zero_rows_indices = zero_rows_indices1 + zero_rows_indices2
pred_array = pred_array[~np.isin(np.arange(pred_array.shape[0]), zero_rows_indices)]
true_array = true_array[~np.isin(np.arange(true_array.shape[0]), zero_rows_indices)]
corr_pred = np.corrcoef(pred.values,dtype=np.float32)
corr_true = np.corrcoef(true.values,dtype=np.float32)
cmd_cell = cal_cmd(corr_pred,corr_true)

pred_array = pred.T.values
true_array = true.T.values
zero_rows_indices1 = list(np.where(~pred_array.any(axis=1))[0])
zero_rows_indices2 = list(np.where(~true_array.any(axis=1))[0])
zero_rows_indices = zero_rows_indices1 + zero_rows_indices2
pred_array = pred_array[~np.isin(np.arange(pred_array.shape[0]), zero_rows_indices)]
true_array = true_array[~np.isin(np.arange(true_array.shape[0]), zero_rows_indices)]
corr_pred = np.corrcoef(pred_array,dtype=np.float32)
corr_true = np.corrcoef(true_array,dtype=np.float32)
cmd_peak = cal_cmd(corr_pred,corr_true)
    
## RMSE 
rmse_all_df = cal_rmse_all(pred,true)

## AUROC
auroc_all_df = cal_auroc_all(pred,true)
