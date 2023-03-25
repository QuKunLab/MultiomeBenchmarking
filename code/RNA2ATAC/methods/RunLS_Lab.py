#!/usr/bin/env python
# coding: utf-8

# In[97]:


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import numpy as np
import sys
import scanpy as sc
import anndata as ad
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from scipy.sparse import csc_matrix
import logging
from torch.utils.data.dataset import Dataset
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.io import mmread, mmwrite
import pandas as pd


# In[98]:


def read_RNA_ATAC(RNA_path, ATAC_path):
    # gene expression
    cell_names = pd.read_csv(RNA_path+'barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids'] 
    X = csr_matrix(mmread(RNA_path+'matrix.mtx').T)
    gene_names = pd.read_csv(RNA_path+'genes.tsv', sep = '\t',  header=None, index_col=None) 
    gene_names.columns =  ['gene_ids'] 
    adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
    adata_RNA.var_names_make_unique()
    if ATAC_path is None:
        return adata_RNA
    # peak information
    cell_names = pd.read_csv(ATAC_path + 'barcodes.tsv', sep = '\t', header=None, index_col=None)
    cell_names.columns =  ['cell_ids'] 
    X = csr_matrix(mmread(ATAC_path + 'matrix.mtx').T)
    peak_name = pd.read_csv(ATAC_path + 'peaks.bed',header=None,index_col=None)
    peak_name.columns = ['peak_ids']
    adata_ATAC  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
    return adata_RNA, adata_ATAC


# In[ ]:


train_id = "Dataset35"
test_id = "Dataset36"


# In[99]:


## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
data_path = "../data/"+ train_id
path_train = "../data/"+ train_id
path_test = "../data/"+ test_id
par = {
    'input_train_mod1': path_train +  "/RNA/",
    'input_train_mod2': path_train +  "/ATAC/",
    'input_test_mod1': path_test +  "/RNA/",
    'input_test_mod2': path_test +  "/ATAC/",
    'output': '../results/KAUST/KAUST_'+ train_id + '_'+ test_id + '.h5ad',
}
meta = { 'functionality_name': 'lslab' }


# In[100]:


## VIASH END

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(50, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 400)
        self.bn2 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400, 50)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)

        x = self.relu(x)
        x = self.fc3(x)

        return x

class Net_res(nn.Module):
    def __init__(self, num_batches):
        super(Net_res, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(50, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.fc4 = nn.Linear(500, 500)
        self.bn4 = nn.BatchNorm1d(500)
        self.fc5 = nn.Linear(500, 50)

        self.btch_classifier = nn.Linear(500, num_batches)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.bn1(x))
        
        x = x + self.fc2(x)
        x = self.relu(self.bn2(x))
        
        x = x + self.fc3(x)
        x = self.relu(self.bn3(x))
        
        btch = self.btch_classifier(x)

        x = x + self.fc4(x)
        x = self.relu(self.bn4(x))
        
        x = self.fc5(x)
        
        return x, btch

class CustomDataset(Dataset):
    def __init__(self, split, X_train, X_val, X_test, y_train, y_val):
        self.split = split

        if self.split == "train":
            self.data = X_train
            self.gt = y_train
        elif self.split == "val":
            self.data = X_val
            self.gt = y_val
        else:
            self.data = X_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.split == "train":
            return self.data[idx], self.gt[idx]
        elif self.split == "val":
            return self.data[idx], 0
        else:
            return self.data[idx]

class CustomDataset_res(Dataset):
    def __init__(self, split, X_train, X_val, X_test, y_train, y_val, batches):
        self.split = split
        self.batches = batches

        if self.split == "train":
            self.data = X_train
            self.gt = y_train
        elif self.split == "val":
            self.data = X_val
            self.gt = y_val
        else:
            self.data = X_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.split == "train":
            return self.data[idx], self.gt[idx], self.batches[idx]
        elif self.split == "val":
            return self.data[idx], 0, 0
        else:
            return self.data[idx]

criterion = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

def train_model(model, optimizer, criterion, dataloaders_dict, reverse, true_test_mod2, input_test_mod1, input_train_mod2, scheduler, num_epochs):
    best_mse = 100
    best_model = 0

    for epoch in range(num_epochs):
        y_pred = []

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, gts in tqdm(dataloaders_dict[phase]):
                inputs = inputs.cuda()
                gts = gts.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    if phase == 'train':
                        loss = criterion(outputs, gts)
                        running_loss += loss.item() * inputs.size(0)
                        loss.backward()
                        optimizer.step()
                    else:
                        y_pred.extend(outputs.cpu().numpy())


            if phase == "train":
                epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            else:
                y_pred = np.array(y_pred)
                y_pred = y_pred @ reverse

                mse = 0

                for i, sample_gt in enumerate(true_test_mod2.X):
                    mse += ((sample_gt.toarray() - y_pred[i])**2).sum()

                mse = mse / (y_pred.shape[0] * y_pred.shape[1])

                print(mse)

                if mse < best_mse:
                    best_model = copy.deepcopy(model)
                    best_mse = mse
    print("Best MSE: ", best_mse)
    
    return best_model

def train_model_res(model, optimizer, criterion, criterion2, dataloaders_dict, reverse, true_test_mod2, input_test_mod1, input_train_mod2, scheduler, num_epochs):
    best_mse = 100
    best_model = 0

    for epoch in range(num_epochs):
        y_pred = []


        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, gts, btch in tqdm(dataloaders_dict[phase]):
                inputs = inputs.cuda()
                gts = gts.cuda()
                btch = btch.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, out_btch = model(inputs)

                    if phase == 'train':
                        loss1 = criterion(outputs, gts)
                        running_loss += loss1.item() * inputs.size(0)
                        loss2 = criterion2(out_btch, btch)

                        loss = 1 / 9 * loss1 + 8 / 9 * loss2
                        loss.backward()
                        optimizer.step()
                    else:
                        y_pred.extend(outputs.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == "val":
                y_pred = np.array(y_pred)
                y_pred = y_pred @ reverse

                mse = 0

                for i, sample_gt in enumerate(true_test_mod2.X):
                    mse += ((sample_gt.toarray() - y_pred[i])**2).sum()

                mse = mse / (y_pred.shape[0] * y_pred.shape[1])

                print(mse**0.5)

                if mse < best_mse:
                    best_model = copy.deepcopy(model)
                    best_mse = mse
    print("Best RMSE: ", best_mse**0.5)
    
    return best_model

def infer_res(model, dataloader, input_test_mod1, input_train_mod1, input_train_mod2, reverse):
    y_pred = []
    model.eval()

    for inputs in tqdm(dataloader):
        inputs = inputs.cuda()

        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)

            y_pred.extend(outputs.cpu().numpy())

    y_pred = np.array(y_pred)
    y_pred = y_pred @ reverse

    return y_pred
    
def infer(model, dataloader, input_test_mod1, input_train_mod1, input_train_mod2, reverse):
    y_pred = []
    model.eval()

    for inputs in tqdm(dataloader):
        inputs = inputs.cuda()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            y_pred.extend(outputs.cpu().numpy())

    y_pred = np.array(y_pred)
    y_pred = y_pred @ reverse

    return y_pred


# In[101]:


logging.info('Reading `h5ad` files...')
input_train_mod1, input_train_mod2 = read_RNA_ATAC(par['input_train_mod1'], par['input_train_mod2'])
input_train_mod1.var['gene_ids'] = input_train_mod1.var_names
input_train_mod1.var['feature_types'] = pd.Categorical(len(input_train_mod1.var_names)*['GEX'])
input_train_mod1.obs['batch'] = pd.Categorical(len(input_train_mod1.obs)*['batch1'])
input_train_mod1.uns = {'dataset_id': 'pbmc_unsorted_3k', 'organism': 'human'}
input_train_mod1.layers['counts'] = input_train_mod1.X.copy()
input_train_mod2.var['gene_ids'] = input_train_mod2.var_names
input_train_mod2.var['feature_types'] = pd.Categorical(len(input_train_mod2.var_names)*['ATAC'])
input_train_mod2.obs['batch'] = pd.Categorical(len(input_train_mod2.obs)*['batch1'])
input_train_mod2.uns = {'dataset_id': 'pbmc_unsorted_3k', 'organism': 'human'}
input_train_mod2.layers['counts'] = input_train_mod2.X.copy()


# In[ ]:


final_input_test_mod1 = read_RNA_ATAC(par['input_test_mod1'], None)
final_input_test_mod1.var['gene_ids'] = final_input_test_mod1.var_names
final_input_test_mod1.var['feature_types'] = pd.Categorical(len(final_input_test_mod1.var_names)*['GEX'])
final_input_test_mod1.obs['batch'] = pd.Categorical(len(final_input_test_mod1.obs)*['batch2'])
final_input_test_mod1.uns = {'dataset_id': 'pbmc_unsorted_3k', 'organism': 'human'}
final_input_test_mod1.layers['counts'] = final_input_test_mod1.X.copy()
input_test_mod2 = read_RNA_ATAC(par['input_test_mod2'],None)
input_test_mod2.var['gene_ids'] = input_test_mod2.var_names
input_test_mod2.var['feature_types'] = pd.Categorical(len(input_test_mod2.var_names)*['ATAC'])
input_test_mod2.obs['batch'] = pd.Categorical(len(input_test_mod2.obs)*['batch2'])
input_test_mod2.uns = {'dataset_id': 'pbmc_unsorted_3k', 'organism': 'human'}
input_test_mod2.layers['counts'] = input_test_mod2.X.copy()


# In[102]:


atac_idx = np.bitwise_and(np.array(input_train_mod2.X.sum(axis=0)>10).flatten(),np.array(input_test_mod2.X.sum(axis=0)>10).flatten())
input_test_mod2 = input_test_mod2[:,atac_idx]
input_train_mod2 = input_train_mod2[:,atac_idx]
input_train_mod2.layers['counts'] = input_train_mod2.X.copy()
input_test_mod2.layers['counts'] = input_test_mod2.X.copy()


# In[103]:


gene_idx = np.bitwise_and(np.array(input_train_mod1.X.sum(axis=0)>10).flatten(),np.array(final_input_test_mod1.X.sum(axis=0)>10).flatten())
final_input_test_mod1 = final_input_test_mod1[:,gene_idx]
input_train_mod1 = input_train_mod1[:,gene_idx]
input_train_mod1.layers['counts'] = input_train_mod1.X.copy()
final_input_test_mod1.layers['counts'] = final_input_test_mod1.X.copy()


# In[105]:


dataset_id = "gex2atac"

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

batches = set(input_train_mod1.obs["batch"])
batch_dict = {batch:i for i, batch in enumerate(batches)}
y = []

for i in range(input_train_mod1.n_obs):
    y.append(int(batch_dict[input_train_mod1.obs["batch"][i]]))

fold = 0

X = input_train_mod1.obs
batches = np.array(y)

inp_train_mod1 = input_train_mod1.copy()
inp_train_mod2 = input_train_mod2.copy()


# In[106]:


params1 = {'learning_rate': 0.3, 
          'depth': 6, 
          'l2_leaf_reg': 3, 
          'loss_function': 'MultiRMSE', 
          'eval_metric': 'MultiRMSE', 
          'task_type': 'CPU', 
          'iterations': 150,
          'od_type': 'Iter', 
          'boosting_type': 'Plain', 
          'bootstrap_type': 'Bernoulli', 
          'allow_const_label': True, 
          'random_state': 1
         }

params2 = {'learning_rate': 0.2, 
          'depth': 7, 
          'l2_leaf_reg': 4, 
          'loss_function': 'MultiRMSE', 
          'eval_metric': 'MultiRMSE', 
          'task_type': 'CPU', 
          'iterations': 200,
          'od_type': 'Iter', 
          'boosting_type': 'Plain', 
          'bootstrap_type': 'Bayesian', 
          'allow_const_label': True, 
          'random_state': 1
         }


# In[107]:


# TODO: implement own method
out1, out2 = 0, 0


# In[108]:


# if "atac2gex" in dataset_id:
#     out_knn = 0
# else: 
#     a=1


# In[109]:


# out_knn


# In[110]:


if "atac2gex" in dataset_id:
    out_knn = 0

    for train_index, test_index in skf.split(X, y):
        print(fold)
        fold += 1

        input_test_mod1 = inp_train_mod1[test_index, :]
        true_test_mod2 = inp_train_mod2[test_index, :]

        input_train_mod1 = inp_train_mod1[train_index, :]
        input_train_mod2 = inp_train_mod2[train_index, :]
    
        input_mod1 = ad.concat(
                {"train": input_train_mod1, "val": input_test_mod1, "test": final_input_test_mod1},
                axis=0,
                join="outer",
                label="group",
                fill_value=0,
                index_unique="-",
            )

        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

        # split dimred back up
        X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        y_train = mod2_pca

        # Get all responses of the training data set to fit the
        # KNN regressor later on.
        # Make sure to use `toarray()` because the output might
        # be sparse and `KNeighborsRegressor` cannot handle it.

        logging.info('Running Linear regression...')
    
        reg = KNeighborsRegressor(n_neighbors=25, metric='minkowski')

        # Train the model on the PCA reduced modality 1 and 2 data
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Project the predictions back to the modality 2 feature space
        y_pred = y_pred @ embedder_mod2.components_
    
        out_knn += y_pred

    y_pred_knn = out_knn / 10
    
    out_rf = 0

    for train_index, test_index in skf.split(X, y):
        print(fold)
        fold += 1

        input_test_mod1 = inp_train_mod1[test_index, :]
        true_test_mod2 = inp_train_mod2[test_index, :]

        input_train_mod1 = inp_train_mod1[train_index, :]
        input_train_mod2 = inp_train_mod2[train_index, :]
    
        input_mod1 = ad.concat(
                {"train": input_train_mod1, "val": input_test_mod1, "test": final_input_test_mod1},
                axis=0,
                join="outer",
                label="group",
                fill_value=0,
                index_unique="-",
            )

        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

        # split dimred back up
        X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        y_train = mod2_pca

        # Get all responses of the training data set to fit the
        # KNN regressor later on.
        # Make sure to use `toarray()` because the output might
        # be sparse and `KNeighborsRegressor` cannot handle it.

        logging.info('Running Linear regression...')
    
        reg = RandomForestRegressor()

        # Train the model on the PCA reduced modality 1 and 2 data
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Project the predictions back to the modality 2 feature space
        y_pred = y_pred @ embedder_mod2.components_
    
        out_rf += y_pred

    y_pred_rf = out_rf / 10
    
    y_pred = 0.45 * y_pred_rf + 0.55 * y_pred_knn
    y_pred = csc_matrix(y_pred)

    adata = ad.AnnData(
        X=y_pred,
       obs=final_input_test_mod1.obs,
       var=inp_train_mod2.var,
       uns={
           'dataset_id': dataset_id,
           'method_id': meta["functionality_name"],
       },
    )
    
    logging.info('Storing annotated data...')
    adata.write_h5ad(par['output'], compression = "gzip")


# In[111]:


if "gex2atac" in dataset_id:
    out_knn = 0

    for train_index, test_index in skf.split(X, y):
        print(fold)
        fold += 1

        input_test_mod1 = inp_train_mod1[test_index, :]
        true_test_mod2 = inp_train_mod2[test_index, :]

        input_train_mod1 = inp_train_mod1[train_index, :]
        input_train_mod2 = inp_train_mod2[train_index, :]
    
        input_mod1 = ad.concat(
                {"train": input_train_mod1, "val": input_test_mod1, "test": final_input_test_mod1},
                axis=0,
                join="outer",
                label="group",
                fill_value=0,
                index_unique="-",
            )

        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

        # split dimred back up
        X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        y_train = mod2_pca

        # Get all responses of the training data set to fit the
        # KNN regressor later on.
        # Make sure to use `toarray()` because the output might
        # be sparse and `KNeighborsRegressor` cannot handle it.

        logging.info('Running Linear regression...')
    
        reg = KNeighborsRegressor(n_neighbors=25, metric='minkowski')

        # Train the model on the PCA reduced modality 1 and 2 data
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Project the predictions back to the modality 2 feature space
        y_pred = y_pred @ embedder_mod2.components_
    
        out_knn += y_pred

    y_pred = out_knn / 10
    y_pred = csc_matrix(y_pred)

    adata = ad.AnnData(
        X=y_pred,
       obs=final_input_test_mod1.obs,
       var=inp_train_mod2.var,
       uns={
           'dataset_id': dataset_id,
           'method_id': meta["functionality_name"],
       },
    )
    
    logging.info('Storing annotated data...')
    adata.write_h5ad(par['output'], compression = "gzip")


# In[113]:


pred=pd.DataFrame(data=adata.X.todense(),index=adata.obs_names,columns=adata.var_names)


# In[115]:


true=pd.DataFrame(data=input_test_mod2.X.todense(),index=input_test_mod2.obs_names,columns=input_test_mod2.var_names)


# In[117]:


pccs_row=[]
for i in range(0,input_test_mod2.n_obs):
    corr = np.corrcoef(true.iloc[i], pred.iloc[i])[0,1]
    pccs_row.append(corr)


# In[118]:


pccs_row_df=pd.DataFrame(data=pccs_row,index=input_test_mod2.obs_names)


# In[119]:


pccs_col=[]
for i in range(0,input_test_mod2.n_vars):
    corr = np.corrcoef(true.iloc[:,i], pred.iloc[:,i])[0,1]
    pccs_col.append(corr)


# In[120]:


pccs_col_df=pd.DataFrame(data=pccs_col,index=input_test_mod2.var_names)


# In[123]:


pccs_row_df.to_csv('../results/KAUST/KAUST_'+ train_id + '_'+ test_id + '_pcc_cell.csv')
pccs_col_df.to_csv('../results/KAUST/KAUST_'+ train_id + '_'+ test_id + '_pcc_peak.csv')


# In[126]:


pred.to_csv('../results/KAUST/KAUST_'+ train_id + '_'+ test_id + '_pred.csv')
true.to_csv('../results/KAUST/KAUST_'+ train_id + '_'+ test_id + '_true.csv')

