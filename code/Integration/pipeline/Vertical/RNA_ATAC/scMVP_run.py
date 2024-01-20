##### formal
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('/public/home/hanyu/Integration/code/scMVP/')
# import dataset
from scMVP.dataset import LoadData,GeneExpressionDataset, CellMeasurement
from scMVP.models import VAE_Attention, Multi_VAE_Attention, VAE_Peak_SelfAttention
from scMVP.inference import UnsupervisedTrainer
from scMVP.inference import MultiPosterior, MultiTrainer
import torch

import scanpy as sc
import anndata

import scipy.io as sp_io
from scipy.sparse import csr_matrix, issparse

import warnings
warnings.filterwarnings("ignore")

arguments = sys.argv

if not os.path.exists(arguments[1]):
    print("Input Data does not exist.")

# 打印解析结果
print("Input file: "+ arguments[1] + " \n")

data_path = arguments[1]

os.chdir(data_path + '/RNA')
os.system('cp features.tsv ../RNA_features.tsv')
os.system('cp barcodes.tsv ../RNA_barcodes.tsv')
os.system('cp matrix.mtx ../RNA_matrix.mtx')
os.chdir(data_path + 'ATAC')
os.system('cp features.tsv ../ATAC_features.tsv')
os.system('cp barcodes.tsv ../ATAC_barcodes.tsv')
os.system('cp matrix.mtx ../ATAC_matrix.mtx')

sciCAR_cellline_dataset = {
                "gene_names": 'RNA_features.tsv',
                "gene_expression": 'RNA_matrix.mtx',
                "gene_barcodes": 'RNA_barcodes.tsv',
                "atac_names": 'ATAC_features.tsv',
                "atac_expression": 'ATAC_matrix.mtx',
                "atac_barcodes": 'ATAC_barcodes.tsv'
                }

dataset = LoadData(dataset=sciCAR_cellline_dataset,data_path=data_path,
                    dense=False,gzipped=False, atac_threshold=0.001,
                    cell_threshold=1)

n_epochs = 10
lr = 1e-3
use_batches = False
use_cuda = True # False if using CPU
n_centroids = 5 
n_alfa = 1.0

multi_vae = Multi_VAE_Attention(dataset.nb_genes, len(dataset.atac_names), n_batch=0, n_latent=20, n_centroids=n_centroids, n_alfa = n_alfa, mode="mm-vae") # should provide ATAC num, alfa, mode and loss type
trainer = MultiTrainer(
    multi_vae,
    dataset,
    train_size=0.9,
    use_cuda=use_cuda,
    frequency=5,
)

trainer.train(n_epochs=n_epochs, lr=lr)
# create posterior from trained model
full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)),type_class=MultiPosterior)
latent, latent_rna, latent_atac, cluster_gamma, cluster_index, batch_indices, labels = full.sequential().get_latent()

np.save(arguments[2]+'/scMVP.npy',latent)