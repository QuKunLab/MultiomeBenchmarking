import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
print(os.getcwd())
os.chdir("./scMOG-main/scMOG_code_Human/data/snareseq_GSE126074/")
print(os.getcwd())
os.system("rm -rf *")
import sys
arguments = sys.argv
input_path = "./scMOG-main/"
data_path =  f'./Dataset37/Train/'
features = pd.read_csv(data_path+"ATAC/features.tsv",header=None)
features = [features[0][i].replace("-",":",1) for i in range(len(features))]
test=pd.DataFrame(columns=['peak'],data=features)
test.to_csv('GSE126074_AdBrainCortex_SNAREseq_chromatin.peaks.tsv',sep='\t',header=None, index=None)
os.system("cp "+data_path+"ATAC/barcodes.tsv GSE126074_AdBrainCortex_SNAREseq_chromatin.barcodes.tsv")
os.system("cp "+data_path+"ATAC/matrix.mtx GSE126074_AdBrainCortex_SNAREseq_chromatin.counts.mtx")
os.system("cp "+data_path+"RNA/barcodes.tsv GSE126074_AdBrainCortex_SNAREseq_cDNA.barcodes.tsv")
os.system("cp "+data_path+"RNA/matrix.mtx GSE126074_AdBrainCortex_SNAREseq_cDNA.counts.mtx")
os.system("cp "+data_path+"RNA/features.tsv GSE126074_AdBrainCortex_SNAREseq_cDNA.genes.tsv")
os.system('gzip *')
os.chdir("./scMOG-main/scMOG_code_Human")
os.system("python bin/Preprocessing.py --snareseq --outdir snareseq_datasets")
os.chdir("./scMOG-main/scMOG_code_Human/snareseq_datasets")
os.system(f"python ../bin/train.py --outdir output --dataset {dataset}")
os.chdir("./scMOG-main/scMOG_code_Human/snareseq_datasets/")
sc_rna_train_dataset=ad.read_h5ad('valid_rna.h5ad')
sc_atac_train_dataset=ad.read_h5ad('valid_atac.h5ad')
path = f'./Dataset37/Test/'
cellinfo = pd.read_csv(path + 'RNA/barcodes.tsv',header=None)
cellinfo.columns = ['cell_id']
geneinfo = pd.read_csv(path + 'RNA/features.tsv',header=None)
geneinfo.columns = ['gene_id']
adata_ing = sc.read(path + 'RNA/matrix.mtx',cache=True).T
adata = sc.AnnData(adata_ing.X, obs=cellinfo ,var=geneinfo)
adata.var_names = adata.var['gene_id']
adata.obs_names = adata.obs['cell_id']
adata.var_names_make_unique()
adata.obs_names_make_unique()
cellinfo = pd.DataFrame(adata.obs_names)
cellinfo.columns = ['cell_id']
geneinfo = pd.DataFrame(adata.var_names)
geneinfo.columns = ['gene_id']
sc_rna_test_dataset = adata
cellinfo = pd.read_csv(path +  'ATAC/barcodes.tsv',header=None)
cellinfo.columns = ['cell_id']
geneinfo = pd.read_csv(path+ 'ATAC/features.tsv',header=None)
geneinfo.columns = ['gene_id']
adata_ing = sc.read(path + 'ATAC/matrix.mtx',cache=True)
sc_atac_test_dataset= sc.AnnData(adata_ing.X.T, obs=cellinfo ,var=geneinfo)
sc_atac_test_dataset.var_names = sc_atac_test_dataset.var['gene_id']
sc_rna_test_dataset.var_names = sc_rna_test_dataset.var['gene_id']
sc_rna_test_dataset[:,sc_rna_train_dataset.var_names].write_h5ad("output/truth_rna_GM.h5ad")
def change_peakname(x):
     ing = x.split("-", 1)
     return ing[0]+":"+ing[1]
sc_atac_test_dataset.var_names = [change_peakname(x) for x in sc_atac_test_dataset.var_names.tolist()]
sc_atac_test_dataset[:,sc_atac_train_dataset.var_names].write_h5ad("output/truth_atac_GM.h5ad")
os.chdir("./scMOG-main/scMOG_code_Human/snareseq_datasets/output")
os.system(f"python ../../bin/predict-atac.py --outdir Otherdataset_generation")
