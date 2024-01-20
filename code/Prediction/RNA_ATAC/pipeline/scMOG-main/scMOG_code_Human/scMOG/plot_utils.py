"""
Helper functions for plotting
"""

import sys
import os
import logging
import random
import warnings
from typing import *
import collections
import itertools

import numpy as np
import pandas as pd
import scipy
import sklearn.metrics as metrics
from sklearn.decomposition import PCA

import mpl_scatter_density
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from adjustText import adjust_text

from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

# mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("seaborn-talk")

try:
    from numba.core.errors import NumbaPerformanceWarning

    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except ModuleNotFoundError:
    pass


from anndata import AnnData
import scanpy as sc

import adata_utils
import utils

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
assert os.path.isdir(SRC_DIR)

SAVEFIG_DPI = 1200

GENE_COLOR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "seurat_like", ["gainsboro", "rebeccapurple"]
)


def get_pca_df(mat, num_pcs=2, group_labels=None):
    """Return a dataframe containing the PCs of the given data"""
    if isinstance(mat, AnnData):
        mat = mat.X
    pca = PCA(n_components=num_pcs)
    pcs = pca.fit_transform(mat)
    pc_df = pd.DataFrame(
        pcs,
        columns=[
            f"PC{i + 1} - {str(round(v, 2))}% variance"
            for i, v in enumerate(pca.explained_variance_)
        ],
    )
    if group_labels is not None:
        pc_df["group"] = group_labels
    return pc_df


def preprocess_anndata(
    a: AnnData,
    neighbors_n_neighbors: int = 15,
    neighbors_n_pcs: Union[int, None] = None,
    louvain_resolution: float = 1.0,
    leiden_resolution: float = 1.0,
    seed: int = 0,
    use_rep: str = None,
    inplace: bool = True,
):
    """
    Preprocess the given anndata object to prepare for plotting. This occurs in place
    Performs dimensionality reduction, projection, and clustering
    """
    assert isinstance(a, AnnData)
    if not inplace:
        raise NotImplementedError
    assert a.shape[0] >= 50, f"Got input with too few dimensions: {a.shape}"
    sc.pp.pca(a)
    # sc.tl.tsne(a, n_jobs=12, use_rep=use_rep)
    sc.tl.tsne(a,use_rep=use_rep)  # Representation defaults to X_pca
    # Representation defaults to X_pca
    sc.pp.neighbors(
        a, use_rep=use_rep, n_neighbors=neighbors_n_neighbors, n_pcs=neighbors_n_pcs
    )  # Representation defaults to X_pca
    # https://rdrr.io/cran/Seurat/man/RunUMAP.html
    sc.tl.umap(  # Does not have a rep, looks at neighbors
        a,
        maxiter=500,
        min_dist=0.3,  # Seurat default is 0.3, scanpy is 0.5
        spread=1.0,  # Seurat default is 1.0
        alpha=1.0,  # Seurat default starting learning rate is 1.0
        gamma=1.0,  # Seurate default repulsion strength is 1.0
        negative_sample_rate=5,  # Seurat default negative sample rate is 5
    )  # Seurat default is 200 for large datasets, 500 for small datasets
    if louvain_resolution > 0:
        sc.tl.louvain(  # Depends on having neighbors or bbknn run first
            adata=a, resolution=louvain_resolution,random_state=seed
        )  # Seurat also uses Louvain
    else:
        logging.info("Skipping louvain clustering")
    if leiden_resolution > 0:
        sc.tl.leiden(  # Depends on having neighbors or bbknn first
            adata=a, resolution=leiden_resolution,random_state=seed, n_iterations=2
        )  # R runs 2 iterations
    else:
        logging.info("Skipping leiden clustering")


def plot_clustering_anndata(
    a: AnnData,
    pca: bool = True,
    tsne: bool = True,
    umap: bool = True,
    color: str = "leiden",
    label_counter: bool = False,
    fname: str = "",
):
    """Plot the given anndata object after it's been preprocessed as above"""
    plotting_funcs = {
        "pca": sc.pl.pca_scatter,
        "tsne": sc.pl.tsne,
        "umap": sc.pl.umap,
    }
    if not pca:
        plotting_funcs.pop("pca")
    if not tsne:
        plotting_funcs.pop("tsne")
    if not umap:
        plotting_funcs.pop("umap")
    num_subplots = len(plotting_funcs)
    assert num_subplots, f"No clustering subplots enabled"

    padding = max(0, num_subplots - 1)
    fig, axes = plt.subplots(
        dpi=300, ncols=num_subplots, figsize=(num_subplots * 5 + padding, 5)
    )
    for i, ((plot_type, plot_func), ax) in enumerate(zip(plotting_funcs.items(), axes)):
        plot_func(a, show=False, ax=ax, alpha=0.8, color=color)
        if ax is not axes[-1]:
            ax.legend([])
        else:
            cnt = collections.Counter(a.obs[color])
            handles, labels = ax.get_legend_handles_labels()
            if label_counter:
                labels = [l + f" (n={cnt[l]})" for l in labels]
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    if fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
    return fig


def plot_clustering_anndata_direct_label(
    a: AnnData,
    color: str,
    representation: str = "umap",
    representation_axes_label: str = "",
    swap_axes: bool = False,
    cmap: Callable = plt.get_cmap("tab20"),
    adjust: bool = False,
    ax_tick: bool = False,
    title: str = "",
    fname: str = "",
    **kwargs,
):
    """
    Plot the given adata's representation, directly labelling instead of using
    a legend
    """
    rep_key = "X_" + representation
    assert (
        rep_key in a.obsm
    ), f"Representation {representation} not fount in keys {a.obsm.keys()}"

    coords = a.obsm[rep_key]
    if swap_axes:
        coords = coords[:, ::-1]  # Reverse the columns
    assert isinstance(coords, np.ndarray) and len(coords.shape) == 2
    assert coords.shape[0] == a.n_obs
    assert color in a.obs
    color_vals = a.obs[color]
    unique_val = np.unique(color_vals.values)
    color_idx = [sorted(list(unique_val)).index(i) for i in color_vals]
    # Vector of colors for each point
    color_vec = [cmap(i) for i in color_idx]

    fig, ax = plt.subplots(figsize=(4.5, 6), dpi=300)
    ax.scatter(
        coords[:, 0], coords[:, 1], s=12000 / coords.shape[0], c=color_vec, alpha=0.9
    )

    # Label each cluster
    texts = []
    for v in unique_val:
        v_idx = np.where(color_vals.values == v)
        # Median behaves better with outliers than mean
        v_coords = np.median(coords[v_idx], axis=0)
        t = ax.text(
            *v_coords,
            v,
            horizontalalignment="center",
            verticalalignment="center",
            size=7,
        )
        texts.append(t)
    if adjust:
        adjust_text(
            texts,
            only_move={"texts": "y"},
            force_text=0.01,
            autoalign="y",
            avoid_points=False,
        )

    rep_str = representation_axes_label if representation_axes_label else representation
    if not swap_axes:
        ax.set(
            xlabel=f"{rep_str.upper()}1",
            ylabel=f"{rep_str.upper()}2",
        )
        ax.set_xlabel(f"{rep_str.upper()}1", fontsize=7)
        ax.set_ylabel(f"{rep_str.upper()}2", fontsize=7)
    else:
        ax.set(
            xlabel=f"{rep_str.upper()}2",
            ylabel=f"{rep_str.upper()}1",
        )
    ax.set(title=title)
    if not ax_tick:
        ax.set(xticks=[], yticks=[])
    plt.show()

    if fname:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_clustering_anndata_gene_color(
    a: AnnData,
    gene: str,
    representation: str = "umap",
    representation_axes_label: str = "",
    ax_tick: bool = False,
    seurat_mode: bool = False,
    cmap=GENE_COLOR_CMAP,
    cbar_pos: List[float] = None,
    title: str = "",
    fname: str = "",
    ax=None,
):
    """
    Plot the given adata's representation, coloring by the given gene
    This is function because scanpy's native function for doing this is borked
    in the version we used for this project
    """
    rep_key = (
        "X_" + representation
        if not seurat_mode
        else representation + "_cell_embeddings"
    )
    if rep_key not in a.obsm:
        raise ValueError(
            f"Representation {representation} not found in obms keys {a.obsm.keys()}"
        )

    coords = a.obsm[rep_key]
    assert isinstance(coords, np.ndarray) and len(coords.shape) == 2

    if gene not in a.var_names:
        raise ValueError(f"Gene {gene} not found!")
    colors = a.obs_vector(gene)

    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(4, 3))
    else:
        fig = None
    scat = ax.scatter(
        coords[:, 0], coords[:, 1], c=colors, cmap=cmap, s=12000 / coords.shape[0]
    )
    if cbar_pos is not None:
        # Left bottom width height
        if fig is None:
            fig = ax.figure
        cax = fig.add_axes(cbar_pos)  # Fractional
        cax = fig.colorbar(
            scat, cax, orientation="horizontal", ticks=[np.min(colors), np.max(colors)]
        )
        cax.ax.tick_params(axis="x", pad=1)
        cax.ax.set_xticklabels(["Min", "Max"], size=10)
    rep_str = representation_axes_label if representation_axes_label else representation
    ax.set(
        xlabel=f"{rep_str.upper()}1",
        ylabel=f"{rep_str.upper()}2",
        title=gene if not title else title,
    )
    if not ax_tick:
        ax.set(xticks=[], yticks=[])
    if fname:
        assert fig is not None, "Cannot provide axis and also save fig"
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
    return fig


def plot_scatter_with_r(
    x: Union[np.ndarray, scipy.sparse.csr_matrix],
    y: Union[np.ndarray, scipy.sparse.csr_matrix],
    color=None,
    subset: int = 0,
    logscale: bool = False,
    density_heatmap: bool = False,
    density_dpi: int = 150,
    density_logstretch: int = 1000,
    title: str = "",
    xlabel: str = "Original norm counts",
    ylabel: str = "Inferred norm counts",
    xlim: Tuple[int, int] = None,
    ylim: Tuple[int, int] = None,
    one_to_one: bool = False,
    corr_func: Callable = scipy.stats.pearsonr,
    figsize: Tuple[float, float] = (7, 5),
    fname: str = "",
    ax=None,
):
    """
    Plot the given x y coordinates, appending Pearsons r
    Setting xlim/ylim will affect both plot and R2 calculation
    In other words, plot view mirrors the range for which correlation is calculated
    """
    assert x.shape == y.shape, f"Mismatched shapes: {x.shape} {y.shape}"
    if color is not None:
        assert color.size == x.size
    if one_to_one and (xlim is not None or ylim is not None):
        assert xlim == ylim
    if xlim:
        keep_idx = utils.ensure_arr((x >= xlim[0]).multiply(x <= xlim[1]))
        x = utils.ensure_arr(x[keep_idx])
        y = utils.ensure_arr(y[keep_idx])
    if ylim:
        keep_idx = utils.ensure_arr((y >= ylim[0]).multiply(x <= xlim[1]))
        x = utils.ensure_arr(x[keep_idx])
        y = utils.ensure_arr(y[keep_idx])
    # x and y may or may not be sparse at this point
    assert x.shape == y.shape
    if subset > 0 and subset < x.size:
        logging.info(f"Subsetting to {subset} points")
        random.seed(1234)
        # Converts flat index to coordinates
        indices = np.unravel_index(
            np.array(random.sample(range(np.product(x.shape)), k=subset)), shape=x.shape
        )
        x = utils.ensure_arr(x[indices])
        y = utils.ensure_arr(y[indices])
        if isinstance(color, (tuple, list, np.ndarray)):
            color = np.array([color[i] for i in indices])

    if logscale:
        x = np.log1p(x)
        y = np.log1p(y)

    # Ensure correct format
    x = utils.ensure_arr(x).flatten()
    y = utils.ensure_arr(y).flatten()
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))

    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    logging.info(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    logging.info(
        f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}"
    )

    if ax is None:
        fig = plt.figure(dpi=300, figsize=figsize)
        if density_heatmap:
            # https://github.com/astrofrog/mpl-scatter-density
            ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        else:
            ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    if density_heatmap:
        norm = None
        if density_logstretch:
            norm = ImageNormalize(
                vmin=0, vmax=100, stretch=LogStretch(a=density_logstretch)
            )
        ax.scatter_density(x, y, dpi=density_dpi, norm=norm, color="tab:blue")
    else:
        ax.scatter(x, y, alpha=0.2, c=color)

    if one_to_one:
        unit = np.linspace(*ax.get_xlim())
        ax.plot(unit, unit, linestyle="--", alpha=0.5, label="$y=x$", color="grey")
        ax.legend()
    ax.set(
        xlabel=xlabel + (" (log)" if logscale else ""),
        ylabel=ylabel + (" (log)" if logscale else ""),
        title=(title + f" ($r={pearson_r:.2f}$)").strip(),
    )
    if xlim:
        ax.set(xlim=xlim)
    if ylim:
        ax.set(ylim=ylim)

    if fig is not None and fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")

    return fig


def plot_bulk_scatter(
    x: AnnData,
    y: AnnData,
    x_subset: dict = None,
    y_subset: dict = None,
    logscale: bool = True,
    corr_func: Callable = scipy.stats.pearsonr,
    title: str = "",
    xlabel: str = "Average measured counts",
    ylabel: str = "Average inferred counts",
    fname: str = "",
):
    """
    Create bulk signature and plot
    """
    if x_subset is not None:
        orig_size = x.n_obs
        x = adata_utils.filter_adata(x, filt_cells=x_subset)
        logging.info(f"Subsetted x from {orig_size} to {x.n_obs}")
    if y_subset is not None:
        orig_size = y.n_obs
        y = adata_utils.filter_adata(y, filt_cells=y_subset)
        logging.info(f"Subsetted y from {orig_size} to {y.n_obs}")

    # Make sure variables match
    shared_var_names = sorted(list(set(x.var_names).intersection(y.var_names)))
    logging.info(f"Found {len(shared_var_names)} shared variables")
    x = x[:, shared_var_names]
    y = y[:, shared_var_names]

    x_vals = x.X
    y_vals = y.X

    if logscale:
        x_vals = np.log1p(x_vals)
        y_vals = np.log1p(y_vals)

    # Ensure correct format
    x_vals = utils.ensure_arr(x_vals).mean(axis=0)
    y_vals = utils.ensure_arr(y_vals).mean(axis=0)

    assert not np.any(np.isnan(x_vals))
    assert not np.any(np.isnan(y_vals))

    pearson_r, pearson_p = corr_func(x_vals, y_vals)
    logging.info(f"Found pearson's correlation of {pearson_r:.4f}")

    fig, ax = plt.subplots(dpi=SAVEFIG_DPI, figsize=(7, 5))
    ax.scatter(x_vals, y_vals, alpha=0.4)
    ax.set(
        xlabel=xlabel + (" (log)" if logscale else ""),
        ylabel=ylabel + (" (log)" if logscale else ""),
        title=(title + f" ($r={pearson_r:.2f}$)").strip(),
    )
    if fname:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_expression_comparison_hist(
    adata: AnnData, gene: str, split_by: str = "IsPost"
):
    """Plot histogram comparing the expression of a single gene across split_by"""
    fig, ax = plt.subplots(dpi=300)
    vals = {}
    for level in sorted(list(set(adata.obs[split_by])))[::-1]:
        level_idx = np.where(adata.obs[split_by] == level)
        v = adata.obs_vector(gene)[level_idx]
        ax.hist(v, label=level, bins=25, alpha=0.7)
        vals[level] = v
    if len(vals) == 2:
        # Do a rank sum test if only 2 categories
        stat, pval = scipy.stats.mannwhitneyu(*list(vals.values()))
        ax.annotate(f"p={pval:.4e}", xy=(0.05, 0.9), xycoords="axes fraction")
    ax.legend()
    ax.set(
        xlabel="RNA Expression",
        title=f"Expression of {gene}, split by {split_by}",
    )
    return fig


def plot_expression_comparison_violin(
    adata: AnnData,
    gene: str,
    split_by: str,
    split_key: Union[str, List[str]] = "",
    title: str = "",
    fname: str = "",
    simplify_names: bool = True,
    diag_labels: bool = False,
    alt_hypothesis: str = "two-sided",
    ax=None,
):
    """Plot violinplot comparing expression of a single gene"""

    def simplify_name(s: str) -> str:
        mapping = {
            "CD4+ T follicular helper": "Tfh",
        }
        if s in mapping:
            return mapping[s]
        return s

    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(4, 3))
    else:
        fig = None
    vals = collections.defaultdict(list)

    for level in sorted(list(set(adata.obs[split_by])))[::-1]:
        level_idx = np.where(adata.obs[split_by] == level)
        v = adata.obs_vector(gene)[level_idx]
        if not split_key:
            vals[level].extend(list(v))
        else:
            if (isinstance(split_key, str) and level == split_key) or (
                isinstance(split_key, (list, tuple, set)) and level in split_key
            ):
                vals[level].extend(list(v))
            else:
                vals["Other"].extend(list(v))

    if simplify_names:
        vals = {simplify_name(k): np.array(v) for k, v in vals.items()}

    vals_keys = sorted(list(vals.keys()))
    vals_keys.remove("Other")
    vals_keys.insert(0, "Other")
    sns.violinplot(
        data=[vals[k] for k in vals_keys],
        inner="quartile",
        ax=ax,
        width=0.9,
        palette="pastel",
    )
    if len(vals) == 2:
        stat, pval = scipy.stats.mannwhitneyu(
            *list(vals.values()), alternative=alt_hypothesis
        )
        logging.info(f"Statistic of {stat:.4g}")
        logging.info(f"hypo {alt_hypothesis} p-value of {pval:.4g}")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(vals_keys, rotation=45 if diag_labels else 0)
    ax.set(title=gene if not title else title)
    if fname:
        assert fig is not None, f"Cannot provide both fname and axes"
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")

    return fig


def plot_var_vs_explained_var(
    adata_truths: AnnData,
    adata_preds: AnnData,
    highlight_genes: Dict[str, List[str]] = None,
    logscale: bool = True,
    constrain_y_axis: bool = True,
    label_outliers: bool = False,
    fname: str = "",
    fname_gene_list: str = "",
):
    """
    Plot variance versus explained variance
    https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score
    """

    def label_if_outlier(
        x: pd.Series, y: pd.Series, ax, low_cutoff=-0.1, high_cutoff=0.3
    ) -> None:
        """Label if outlier according to the defined cutoffs"""
        for idx in y.index:
            if y[idx] < low_cutoff:
                logging.info(f"Low gene: {idx}")
                ax.annotate(idx, (x[idx], y[idx]))
            elif y[idx] > high_cutoff:
                logging.info(f"High gene: {idx}")
                ax.annotate(idx, (x[idx], y[idx]))

    per_gene_variance = pd.Series(
        np.var(adata_truths.X, axis=0), index=adata_truths.var_names
    )

    explained_var = pd.Series(
        [
            metrics.explained_variance_score(adata_truths.X[:, i], adata_preds.X[:, i])
            for i in range(adata_truths.X.shape[1])
        ],
        index=adata_preds.var_names,
    )
    logging.info(
        f"Explained variance per gene mean/sd/median: {np.mean(explained_var)} {np.std(explained_var)} {np.median(explained_var)}"
    )

    assert per_gene_variance.shape == explained_var.shape

    # If we want to output to file, write it
    if fname_gene_list:
        # Find top N and write it out
        idx_sort = np.argsort(explained_var)
        lowest_genes = list(explained_var[idx_sort[:25]].index)
        highest_genes = list(explained_var[idx_sort[-25:]].index)

        with open(fname_gene_list, "w") as sink:
            sink.write("Low\n")
            sink.write("\n".join(lowest_genes) + "\n")
            sink.write("High\n")
            sink.write("\n".join(highest_genes) + "\n")

    nonhighlight_genes = list(per_gene_variance.index)
    if highlight_genes:
        all_highlighted_genes = set(
            itertools.chain.from_iterable(highlight_genes.values())
        )
        # Filter out each set of genes that awe are highlighting
        nonhighlight_genes = [
            g for g in per_gene_variance.index if g not in all_highlighted_genes
        ]

    fig, ax = plt.subplots(dpi=300)
    if logscale:
        ax.set_xscale("symlog")
    ax.scatter(
        per_gene_variance[nonhighlight_genes],
        explained_var[nonhighlight_genes],
        alpha=0.8,
    )
    if label_outliers:
        label_if_outlier(
            per_gene_variance[nonhighlight_genes],
            explained_var[nonhighlight_genes],
            ax=ax,
        )
    if highlight_genes:
        for key, genes in highlight_genes.items():
            l = len(set(genes).intersection(per_gene_variance.index))
            logging.info(f"Plotting {l} genes under {key}")
            var_sub = per_gene_variance[genes]
            explained_sub = explained_var[genes]
            ax.scatter(
                var_sub,
                explained_sub,
                alpha=0.8,
                label=key.strip() + f" (n={l})",
            )
            if label_outliers:
                label_if_outlier(var_sub, explained_sub, ax=ax)
    if constrain_y_axis:
        y_bottom, y_top = ax.get_ylim()
        ax.set(ylim=(max(-3, y_bottom), min(1, y_top)))
    if highlight_genes:
        ax.legend()
    ax.set(
        xlabel="Variance per gene",
        ylabel=f"Explained variance per gene (mean {np.mean(explained_var):.4f}, sd {np.std(explained_var):.4f})",
        title=f"Per-gene variance versus explained variance (n={per_gene_variance.size})",
    )
    if fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI)
    return fig


def plot_binary(
    truth, preds, subset: int = 0, color=None, title="", xlabel="True", fname: str = ""
):
    """
    Given categorical truth, plot what is predicted
    """
    assert truth.shape == preds.shape
    if len(np.unique(truth)) != 2:
        logging.warn(
            f"truth should be binary but got {len(np.unique(truth))}: {np.unique(truth)}"
        )
    if subset:
        assert len(truth.shape) == 1, "Can only subset one-dimensional"
        random.seed(1234)
        idx = np.array(random.sample(range(truth.size), k=subset))
        truth = truth[idx]
        preds = preds[idx]

    fig, ax = plt.subplots(dpi=300)
    zeros = preds[np.where(truth == 0)].flatten()
    ones = preds[np.where(truth == 1)].flatten()
    zeros_sub = np.random.choice(zeros, size=ones.shape, replace=False)
    kld = scipy.special.kl_div(zeros_sub, ones)
    ks, ks_p = scipy.stats.ks_2samp(zeros, ones)
    ax.violinplot([zeros, ones], [0, 1])

    title = " ".join([title, f"(KS={ks:.4f}, p={ks_p:.4E})"])
    ax.set(title=title)
    if fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
    return fig


def dropout_correlation_figure(undropped_truth, dropped_input, denoised, subsample=1e6):
    """
    Given undropped ground truth and denoised output look at correlation
    Subsample controls when we subsample for plotting (r2 is still based on full counts)
    """

    def add_diagonal_and_rescale(ax):
        """Adds diagonal line y=x to the plot and rescales to have axes match"""
        # https://stackoverflow.com/questions/25497402/adding-y-x-to-a-matplotlib-scatter-plot-if-i-havent-kept-track-of-all-the-data
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, color="black")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    assert (
        undropped_truth.shape == denoised.shape == dropped_input.shape
    ), f"Got mismatched shapes: {undropped_truth.shape} {denoised.shape} {dropped_input.shape}"

    undropped_truth = undropped_truth.flatten()
    dropped_input = dropped_input.flatten()
    denoised = denoised.flatten()

    dropped_idx = np.where(dropped_input == 0)
    undropped_truth_drop = undropped_truth[
        dropped_idx
    ]  # "_drop" suffix indicates items that were dropped out
    denoised_drop = denoised[dropped_idx]

    zero_idx = np.where(undropped_truth == 0)
    denoised_zeros = denoised[zero_idx]

    global_correlation = metrics.r2_score(undropped_truth, denoised)
    dropped_correlation = metrics.r2_score(undropped_truth_drop, denoised_drop)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, dpi=300, figsize=(18, 5))
    # Plot all items
    if len(undropped_truth) > subsample:
        rand_idx = np.random.choice(
            np.arange(len(undropped_truth)), size=int(subsample), replace=False
        )
        assert len(undropped_truth) == len(denoised)
        undropped_truth_sub = undropped_truth[rand_idx]
        denoised_sub = denoised[rand_idx]
        ax1.scatter(undropped_truth_sub, denoised_sub, alpha=0.7)
    else:
        ax1.scatter(undropped_truth, denoised, alpha=0.7)
    add_diagonal_and_rescale(ax1)
    ax1.set(
        title=f"True vs imputed counts (r2={global_correlation:.4f})",
        xlabel="True count (pre-dropout)",
        ylabel="Imputed count",
    )

    # Plot items that were dropped out
    if len(undropped_truth_drop) > subsample:
        rand_idx = np.random.choice(
            np.arange(undropped_truth_drop.size), size=int(subsample), replace=False
        )
        undropped_truth_drop = undropped_truth_drop[rand_idx]
        denoised_drop = denoised_drop[rand_idx]
    ax2.scatter(undropped_truth_drop, denoised_drop, alpha=0.7)
    add_diagonal_and_rescale(ax2)
    ax2.set(
        title=f"True vs imputed counts (dropout genes, r2={dropped_correlation:.4f})",
        xlabel="True count (pre-dropout)",
        ylabel="Imputed count",
    )

    # Plot items that are supposed to be 0
    if len(denoised_zeros) > subsample:
        rand_idx = np.random.choice(
            np.arange(denoised_zeros.size), size=int(subsample), replace=False
        )
        denoised_zeros = denoised_zeros[rand_idx]
    ax3.hist(denoised_zeros, bins=35, alpha=0.8)
    ax3.set(
        title=f"Imputation of true 0s\n(median={np.median(denoised_zeros):.4f}, mean={np.mean(denoised_zeros):.4f}, %0={np.sum(denoised_zeros == 0) / denoised_zeros.size})",
    )

    return fig


def plot_latent_difference(
    latent1, latent2, flatten=True, max_val=1.0, nbins=35, fname=None
):
    """
    Plot the difference in latent space
    """
    delta = latent1 - latent2
    if not isinstance(delta, np.ndarray):
        delta = delta.detach().cpu().numpy()

    assert len(delta.shape) <= 2

    if flatten:
        latent1 = latent1.flatten()
        latent2 = latent2.flatten()
        delta = delta.flatten()

    if np.any(latent1 < 0) or np.any(latent2 < 0):
        bins = np.linspace(-max_val, max_val, num=nbins)
        bins_symmetric = bins
    else:
        bins = np.linspace(0, max_val, num=nbins)
        bins_symmetric = np.concatenate((-bins[:0:-1], bins))

    fig, (ax1, ax2, ax3) = plt.subplots(dpi=300, ncols=3, figsize=(15, 4))
    ax1.hist(latent1, bins=bins, alpha=0.75)
    ax1.set(
        title="Latent 1",
    )
    ax2.hist(latent2, bins=bins, alpha=0.75)
    ax2.set(
        title="Latent 2",
    )
    ax3.hist(delta, bins=bins_symmetric, alpha=0.75)
    ax3.set(
        title="Difference in encoded representation",
    )

    if fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
    else:
        return fig


def plot_heatmap(adata: AnnData, features, groupby=None):
    """
    Plot a heatmap of the matrix using the given features
    """
    assert isinstance(adata, AnnData)
    valid_features = [ft for ft in features if ft in adata.var_names]
    for ft in features:
        if ft not in valid_features:
            logging.warn(f"Feature not found: {ft}")

    sc.pl.heatmap(
        adata=adata,
        var_names=valid_features,
        groupby=groupby,
    )


def plot_auroc(
    truth,
    preds,
    title_prefix: str = "Receiver operating characteristic",
    fname: str = "",
):
    """
    Plot AUROC after flattening inputs
    """
    truth = utils.ensure_arr(truth).flatten()
    preds = utils.ensure_arr(preds).flatten()
    fpr, tpr, _thresholds = metrics.roc_curve(truth, preds)
    auc = metrics.auc(fpr, tpr)
    logging.info(f"Found AUROC of {auc:.4f}")

    fig, ax = plt.subplots(dpi=300, figsize=(7, 5))
    ax.plot(fpr, tpr)
    ax.set(
        xlim=(0, 1.0),
        ylim=(0.0, 1.05),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"{title_prefix} (AUROC={auc:.2f})",
    )
    if fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
    return fig


def plot_auprc(
    truth,
    preds,
    title_prefix: str = "Precision recall curve",
    fname: str = "",
):
    """Plot AUPRC"""
    truth = utils.ensure_arr(truth).flatten()
    preds = utils.ensure_arr(preds).flatten()

    precision, recall, _thresholds = metrics.precision_recall_curve(truth, preds)
    average_precision = metrics.average_precision_score(truth, preds)
    logging.info(f"Found AUPRC of {average_precision:.4f}")

    fig, ax = plt.subplots(dpi=SAVEFIG_DPI, figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"{title_prefix} (AUPRC={average_precision:.4f})",
    )
    if fname:
        fig.savefig(fname, bbox_inches="tight")
    return fig
