import os, sys
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import scipy
from scipy import sparse
import importlib

import anndata as ad
from scipy.io import mmread, mmwrite

import matplotlib.pyplot as plt
import seaborn as sns

import scib
import scib_metrics
from scib_metrics.benchmark import Benchmarker

from typing import Any, Callable, Optional, Union
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
import matplotlib
arguments = sys.argv

def PlotTable(df,num_embeds = 7, min_max_scale: bool = True, show: bool = True, save_dir: Optional[str] = None):
    # num_embeds = len(self._embedding_obsm_keys)
    # num_embeds = len(self._embedding_obsm_keys)
    cmap_fn = lambda col_data: normed_cmap(col_data, cmap=matplotlib.cm.PRGn, num_stds=2.5)
        # df = bm.get_results(min_max_scale=min_max_scale)
        # Do not want to plot what kind of metric it is
    plot_df = df.drop('Metric Type', axis=0)
    plot_df = plot_df.astype(np.float64)
        # Sort by total score
    plot_df = plot_df.sort_values(by="Total", ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

        # Split columns by metric type, using df as it doesn't have the new method col
    score_cols = df.columns[df.loc['Metric Type'] == 'Aggregate score']
    other_cols = df.columns[df.loc['Metric Type'] != 'Aggregate score']
    column_definitions = [
        ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
    ]
        # Circles for the metric values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=cmap_fn(plot_df[col]),
            group=df.loc['Metric Type', col],
            formatter="{:.2f}",
        )
        for i, col in enumerate(other_cols)
    ]
        # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": matplotlib.cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=df.loc['Metric Type', col],
            border="left" if i == 0 else None,
        )
        for i, col in enumerate(score_cols)
    ]
        # Allow to manipulate text post-hoc (in illustrator)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)
    if show:
        plt.show()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "scib_results.png"), facecolor=ax.get_facecolor(), dpi=300)
        fig.savefig(os.path.join(save_dir, "scib_results.pdf"), facecolor=ax.get_facecolor(), dpi=300, bbox_inches="tight")

    return tab

method_list = ['Multigrate','scMoMaT','scVAEIT','StabMap','UINMF','totalVI','scArches'] ## the methods you want to compute the metrics for whose results

results_path = "./Results/" ## path to integration latent results
data_path = "./dataset/" ## path to raw data and metadata
save_path = "./Results/"  ## path to save the metrics results

for method in method_list:
    print(method)
    locals()[method+"_data"] = pd.read_csv(results_path+method+".csv",header=None)
    print(eval(method+"_data").shape)

path = data_path+"/batch1"
# gene expression
cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
gene_names = pd.read_csv(path+'/RNA/features.tsv', sep = '\t',  header=None, index_col=None)
gene_names.columns =  ['gene_ids']
adata_RNA = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = gene_names.gene_ids))
adata_RNA.var_names_make_unique()

path = data_path+"/batch2"
## scRNA-seq
cell_names = pd.read_csv(path+'/RNA/barcodes.tsv', sep = '\t', header=None, index_col=None)
cell_names.columns =  ['cell_ids']
X = csr_matrix(mmread(path+'/RNA/matrix.mtx').T)
peak_name = pd.read_csv(path+'/RNA/features.tsv',header=None,index_col=None)
peak_name.columns = ['peak_ids']
adata_rna  = ad.AnnData(X, obs=pd.DataFrame(index=cell_names.cell_ids), var=pd.DataFrame(index = peak_name.peak_ids))
adata_rna.var['modality'] = ['Gene Expression']*adata_rna.shape[1]
del X
# # We can now use the organizing method from scvi to concatenate these anndata
sc.pp.filter_cells(adata_RNA, min_genes=1)
sc.pp.filter_genes(adata_RNA, min_cells=20)
sc.pp.filter_cells(adata_rna, min_genes=1)
sc.pp.filter_genes(adata_rna, min_cells=20)

adata_RNA.obs_names = ['batch1_' + item for item in adata_RNA.obs_names]
adata_rna.obs_names = ['batch2_' + item for item in adata_rna.obs_names]

adata = sc.concat([adata_RNA, adata_rna],axis=0)
adata.obs['batch'] = adata_RNA.shape[0]*['batch1'] + adata_rna.shape[0]*['batch2']
del adata_RNA, adata_rna

for method in method_list:
    print(method)
    locals()[method+"_data"].index = adata.obs_names
    adata.obsm[method] = eval(method+"_data")

for i in range(1,3):
    locals()['batch'+str(i)+'_meta'] = pd.read_csv(data_path+'batch'+str(i)+'_celltype.csv',index_col=0)
    locals()['batch'+str(i)+'_meta'].index = ['batch'+str(i)+'_' + item for item in eval('batch'+str(i)+'_meta').index]

meta = pd.DataFrame()
for i in range(1,3):
    meta = pd.concat([meta,eval('batch'+str(i)+'_meta')])

meta = meta[~meta.index.duplicated()]

inter_cell = meta.index.intersection(adata.obs_names)
adata = adata[inter_cell,:]
meta = meta.T[inter_cell].T

adata.obs['celltype'] = meta['celltype']

bm = Benchmarker(
    adata,
    batch_key="batch",
    label_key="celltype",
    embedding_obsm_keys=method_list,
    n_jobs=6,
)
bm.benchmark()

for method in method_list:
    print(method)
    sc.pp.neighbors(adata, use_rep=method)
    scib.me.cluster_optimal_resolution(adata, cluster_key="cluster_"+method, label_key="celltype")
    locals()[method+'_ari'] = scib.me.ari(adata, cluster_key="cluster_"+method, label_key="celltype")
    locals()[method+'_nmi'] = scib.me.nmi(adata, cluster_key="cluster_"+method, label_key="celltype")
    bm._results[method]['nmi_ari_cluster_labels_kmeans_nmi'] = eval(method+'_nmi')
    bm._results[method]['nmi_ari_cluster_labels_kmeans_ari'] = eval(method+'_ari')

df = bm.get_results(min_max_scale=False)

df.to_csv(save_path+"/metrics.csv")

PlotTable(df,num_embeds = len(method_list), min_max_scale=False,save_dir = save_path+"/")

