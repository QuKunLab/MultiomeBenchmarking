"""
Utility functions for anndata objects
"""

import os
import sys
import logging
from typing import *
import multiprocessing
import functools
import collections
import itertools
import re

os.environ["NUMEXPR_MAX_THREADS"] = "32"

import tqdm

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from statsmodels.stats import multitest
import sklearn
from scipy import io

import anndata as ad
from anndata import AnnData

from genomic_interval import GenomicInterval
import utils

logging.basicConfig(level=logging.INFO)


def write_adata_as_10x_dir(
    x: AnnData, outdir: str, transpose: bool = True, mode: str = "ATAC"
) -> None:
    """Write the given anndata object as a directory"""
    if not os.path.isdir(outdir):
        logging.info(f"Creating output directory: {outdir}")
        os.makedirs(outdir)

    # Write output
    logging.info("Writing mtx sparse matrix")
    io.mmwrite(os.path.join(outdir, "matrix.mtx"), x.X if not transpose else x.X.T)

    # Write metadata
    logging.info(f"Writing variable metadata")
    if mode == "ATAC":
        with open(os.path.join(outdir, "peaks.bed"), "w") as sink:
            for gi_str in x.var_names:
                gi = GenomicInterval(gi_str)
                gi_tokens = map(str, gi.as_tuple())
                sink.write("\t".join(gi_tokens) + "\n")
        matrix = pd.DataFrame(x.var_names)
        matrix.to_csv(os.path.join(outdir, 'features.tsv'), sep="\t", index=False,header=None)
    else:
        # raise NotImplementedError

        matrix = pd.DataFrame(x.var_names)
        matrix.to_csv(os.path.join(outdir, 'features.tsv'), sep="\t", index=False,header=None)


    # Write the barcodes
    logging.info(f"Writing cell metadata")
    with open(os.path.join(outdir, "barcodes.tsv"), "w") as sink:
        for n in x.obs_names:
            sink.write(n + "\n")


def annotate_basic_adata_metrics(adata: AnnData) -> None:
    """Annotate with some basic metrics"""
    assert isinstance(adata, AnnData)
    adata.obs["n_counts"] = np.squeeze(np.asarray((adata.X.sum(1))))
    adata.obs["log1p_counts"] = np.log1p(adata.obs["n_counts"])
    adata.obs["n_genes"] = np.squeeze(np.asarray(((adata.X > 0).sum(1))))

    adata.var["n_counts"] = np.squeeze(np.asarray(adata.X.sum(0)))
    adata.var["log1p_counts"] = np.log1p(adata.var["n_counts"])
    adata.var["n_cells"] = np.squeeze(np.asarray((adata.X > 0).sum(0)))


def merge_adata(
    adatas: List[AnnData],
    mask_vals: bool = True,
    max_var_prop: float = 1.0,  # Allowable variance as a proportion of the gene's value
    reduce_func: Callable = np.median,
) -> AnnData:
    """Given adatas with the same obs/var combinations, combine them"""
    # Ensure all are the same shape
    logging.info(f"Aggregating across {len(adatas)} anndata objects")
    assert len(set([a.shape for a in adatas])) == 1
    if len(adatas) == 1:
        return adatas.pop()
    ret_mat = np.zeros(adatas[0].shape)
    var_mat = np.zeros(adatas[0].shape)

    # Process each row at a time to save memory
    for i in range(adatas[0].shape[0]):
        row_stacked = np.vstack([a.X[i] for a in adatas])
        row_variance = np.var(row_stacked, axis=0)
        var_mat[i] = row_variance
        row_reduced = reduce_func(row_stacked, axis=0)
        ret_mat[i] = row_reduced
    obs_df = adatas[0].obs
    var_df = adatas[0].var
    if mask_vals:
        assert (
            0.0 < max_var_prop <= 1.0
        ), f"Max var prop value out of allowed range: {max_var_prop}"
        mask = np.zeros_like(ret_mat, dtype=bool)
        # Elements where variance is greater than returned value
        mask = np.logical_or(var_mat > (max_var_prop * ret_mat), mask)
        # Drop the genes where we have questionable predictions
        mask_vars = np.any(mask, axis=0)
        logging.info(f"Dropping {np.sum(mask_vars)} features")
        ret_mat = np.delete(ret_mat, np.where(mask_vars), axis=1)
        var_df.drop(index=var_df.index[np.where(mask_vars)], inplace=True)
    retval = AnnData(ret_mat, obs=obs_df, var=var_df)
    return retval


def check_marker_genes(
    adata: AnnData,
    marker_genes: List[str],
    groupby: str = "leiden",
    use_raw: bool = False,
    fdr_correction: bool = True,
    filt: bool = True,
) -> pd.DataFrame:
    """
    Check if the given marker genes are differentially expressed
    Anndata object must have groupby key present
    """
    # Establish a matrix of marker_genes x cluster_ids
    cluster_ids = sorted(list(set(adata.obs[groupby])))
    # Filter out genes so that we only include those actually observed
    marker_genes = [m for m in marker_genes if m in adata.var_names]
    logging.info(
        f"Checking differential expression of {len(marker_genes)} genes across {len(cluster_ids)} clusters"
    )
    precorrection = pd.DataFrame(index=marker_genes, columns=cluster_ids)
    for gene in marker_genes:
        for cluster_id in cluster_ids:
            this_cluster_cell_idx = np.where(adata.obs[groupby] == cluster_id)
            other_cluster_cells_idx = np.where(adata.obs[groupby] != cluster_id)
            this_cluster_gene_vals = adata.obs_vector(gene)[this_cluster_cell_idx]
            other_cluster_gene_vals = adata.obs_vector(gene)[other_cluster_cells_idx]
            assert (
                this_cluster_gene_vals.size + other_cluster_gene_vals.size
                == adata.n_obs
            )
            stat, pval = scipy.stats.ranksums(
                this_cluster_gene_vals,
                other_cluster_gene_vals,
            )
            precorrection.loc[gene, cluster_id] = pval
    s = precorrection.shape
    corrected = pd.DataFrame(
        multitest.multipletests(precorrection.values.flatten())[1].reshape(s),
        index=precorrection.index,
        columns=precorrection.columns,
    )
    return corrected if fdr_correction else precorrection


def find_marker_genes(
    adata: AnnData,
    groupby: str = "leiden",
    n_genes: int = 25,
    use_raw: bool = False,
    filt: bool = True,
    method: str = "wilcoxon",
    save_to_uns_key: str = "",
) -> None:
    """
    Find marker genes, recording in adata.uns under
    - rank_genes_groups (may include insignificant genes)
    - optionally, rank_genes_groups_filtered
    """
    # Remove old records if they exist
    if not save_to_uns_key:
        save_to_uns_key = f"rank_genes_{groupby}"
    adata.uns.pop(save_to_uns_key, None)
    adata.uns.pop(save_to_uns_key + "_filtered", None)
    logging.info(f"Saving marker genes to key: {save_to_uns_key}")
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        n_genes=n_genes,
        use_raw=use_raw,
        method=method,
        corr_method="benjamini-hochberg",
        key_added=save_to_uns_key,
    )
    if filt:
        sc.tl.filter_rank_genes_groups(
            adata,
            key=save_to_uns_key,
            groupby=groupby,
            min_fold_change=1.2,
            min_in_group_fraction=0.1,
            max_out_group_fraction=0.9,
            use_raw=use_raw,
            key_added=save_to_uns_key + "_filtered",
        )


def filter_adata(
    adata: AnnData,
    filt_cells: Dict[str, List[str]] = {},
    filt_var: Dict[str, List[str]] = {},
) -> AnnData:
    """
    Filter the AnnData by the given requirements, filtering by cells first then var
    This is based on metadata and returns a copy
    """
    if filt_cells:
        keep_idx = np.ones(adata.n_obs)
        for k, accepted_values in filt_cells.items():
            assert k in adata.obs or k == "index"
            keys = adata.obs[k] if k != "index" else adata.obs.index

            if isinstance(accepted_values, str):
                is_acceptable = np.array([keys == accepted_values])
            elif isinstance(accepted_values, re.Pattern):
                is_acceptable = np.array(
                    [re.search(accepted_values, x) is not None for x in keys]
                )
            elif isinstance(accepted_values, (list, tuple, set, pd.Index)):
                is_acceptable = np.array([x in accepted_values for x in keys])
            else:
                raise TypeError(f"Cannot subset cells using {type(accepted_values)}")
            keep_idx = np.logical_and(is_acceptable.flatten(), keep_idx)
            logging.info(
                f"Filtering cells by {k} retains {np.sum(keep_idx)}/{adata.n_obs}"
            )
        adata = adata[keep_idx].copy()

    if filt_var:
        keep_idx = np.ones(len(adata.var_names))
        for k, accepted_values in filt_var.items():
            assert k in adata.var or k == "index"
            keys = adata.var[k] if k != "index" else adata.var.index

            if isinstance(accepted_values, str):
                is_acceptable = np.array([keys == accepted_values])
            elif isinstance(accepted_values, re.Pattern):
                is_acceptable = np.array(
                    [re.search(accepted_values, x) is not None for x in keys]
                )
            elif isinstance(accepted_values, (list, tuple, set, pd.Index)):
                is_acceptable = np.array([x in accepted_values for x in keys])
            elif isinstance(accepted_values, GenomicInterval):
                is_acceptable = np.array([accepted_values.overlaps(x) for x in keys])
            else:
                raise TypeError(f"Cannot subset features using {type(accepted_values)}")
            keep_idx = np.logical_and(keep_idx, is_acceptable.flatten())
            logging.info(
                f"Filtering vars by {k} retains {np.sum(keep_idx)}/{len(adata.var_names)}"
            )
        adata = adata[:, keep_idx].copy()

    return adata


def filter_adata_cells_and_genes(
    x: AnnData,
    filter_cell_min_counts=None,
    filter_cell_max_counts=None,
    filter_cell_min_genes=None,
    filter_cell_max_genes=None,
    filter_gene_min_counts=None,
    filter_gene_max_counts=None,
    filter_gene_min_cells=None,
    filter_gene_max_cells=None,
) -> None:
    """Filter the count table in place given the parameters based on actual data"""
    args = locals()
    filtering_cells = any(
        [args[arg] is not None for arg in args if arg.startswith("filter_cell")]
    )
    filtering_genes = any(
        [args[arg] is not None for arg in args if arg.startswith("filter_gene")]
    )

    def ensure_count(value, max_value) -> int:
        """Ensure that the value is a count, optionally scaling to be so"""
        if value is None:
            return value  # Pass through None
        retval = value
        if isinstance(value, float):
            assert 0.0 < value < 1.0
            retval = int(round(value * max_value))
        assert isinstance(retval, int)
        return retval

    assert isinstance(x, AnnData)
    # Perform filtering on cells
    if filtering_cells:
        logging.info(f"Filtering {x.n_obs} cells")
    if filter_cell_min_counts is not None:
        sc.pp.filter_cells(
            x,
            min_counts=ensure_count(
                filter_cell_min_counts, max_value=np.max(x.obs["n_counts"])
            ),
        )
        logging.info(f"Remaining cells after min count: {x.n_obs}")
    if filter_cell_max_counts is not None:
        sc.pp.filter_cells(
            x,
            max_counts=ensure_count(
                filter_cell_max_counts, max_value=np.max(x.obs["n_counts"])
            ),
        )
        logging.info(f"Remaining cells after max count: {x.n_obs}")
    if filter_cell_min_genes is not None:
        sc.pp.filter_cells(
            x,
            min_genes=ensure_count(
                filter_cell_min_genes, max_value=np.max(x.obs["n_genes"])
            ),
        )
        logging.info(f"Remaining cells after min genes: {x.n_obs}")
    if filter_cell_max_genes is not None:
        sc.pp.filter_cells(
            x,
            max_genes=ensure_count(
                filter_cell_max_genes, max_value=np.max(x.obs["n_genes"])
            ),
        )
        logging.info(f"Remaining cells after max genes: {x.n_obs}")

    # Perform filtering on genes
    if filtering_genes:
        logging.info(f"Filtering {x.n_vars} vars")
    if filter_gene_min_counts is not None:
        sc.pp.filter_genes(
            x,
            min_counts=ensure_count(
                filter_gene_min_counts, max_value=np.max(x.var["n_counts"])
            ),
        )
        logging.info(f"Remaining vars after min count: {x.n_vars}")
    if filter_gene_max_counts is not None:
        sc.pp.filter_genes(
            x,
            max_counts=ensure_count(
                filter_gene_max_counts, max_value=np.max(x.var["n_counts"])
            ),
        )
        logging.info(f"Remaining vars after max count: {x.n_vars}")
    if filter_gene_min_cells is not None:
        sc.pp.filter_genes(
            x,
            min_cells=ensure_count(
                filter_gene_min_cells, max_value=np.max(x.var["n_cells"])
            ),
        )
        logging.info(f"Remaining vars after min cells: {x.n_vars}")
    if filter_gene_max_cells is not None:
        sc.pp.filter_genes(
            x,
            max_cells=ensure_count(
                filter_gene_max_cells, max_value=np.max(x.var["n_cells"])
            ),
        )
        logging.info(f"Remaining vars after max cells: {x.n_vars}")


def normalize_count_table(
    x: AnnData,
    size_factors: bool = True,
    log_trans: bool = True,
    normalize: bool = True,
) -> AnnData:
    """
    Normalize the count table using method described in DCA paper, performing operations IN PLACE
    rows correspond to cells, columns correspond to genes (n_obs x n_vars)
    s_i is the size factor per cell, total number of counts per cell divided by median of total counts per cell
    x_norm = zscore(log(diag(s_i)^-1 X + 1))

    Reference:
    https://github.com/theislab/dca/blob/master/dca/io.py

    size_factors - calculate and normalize by size factors
    top_n - retain only the top n features with largest variance after size factor normalization
    normalize - zero mean and unit variance
    log_trans - log1p scale data
    """
    assert isinstance(x, AnnData)
    if log_trans or size_factors or normalize:
        x.raw = x.copy()  # Store the original counts as .raw
    # else:
    #     x.raw = x
    if size_factors:
        logging.info("Computing size factors")
        n_counts = np.squeeze(
            np.array(x.X.sum(axis=1))
        )  # Number of total counts per cell
        # Normalizes each cell to total count equal to the median of total counts pre-normalization
        sc.pp.normalize_total(x, inplace=True)
        # The normalized values multiplied by the size factors give the original counts
        x.obs["size_factors"] = n_counts / np.median(n_counts)
        x.uns["median_counts"] = np.median(n_counts)
        logging.info(f"Found median counts of {x.uns['median_counts']}")
        logging.info(f"Found maximum counts of {np.max(n_counts)}")
    else:
        x.obs["size_factors"] = 1.0
        x.uns["median_counts"] = 1.0

    if log_trans:  # Natural logrithm
        logging.info("Log transforming data")
        sc.pp.log1p(
            x,
            chunked=True,
            copy=False,
            chunk_size=100000,
        )

    if normalize:
        logging.info("Normalizing data to zero mean unit variance")
        sc.pp.scale(x, zero_center=True, copy=False)

    return x


def compare_gene_expression(
    adata: AnnData,
    filt_cells: Dict[str, List[str]],
    groupby: str,
    keyadded: str,
    method: str = "wilcoxon",
    n_genes: int = 50,
) -> AnnData:
    """
    Filter the adata by the given dictionary and find marker genes for split
    given by groupby

    Example usage:
    b_cell_adata_prepost_comparison = adata_utils.compare_gene_expression(
        bcc_rna_preds_log,
        filt_cells={
            "ClustersAnnot": ["B", "Plasma B"],
            "leiden_celltypes_expanded_markers": ["B_cells_1"],
        },
        groupby="IsPost",
        keyadded="rank_genes_b_cell_prepost_comparison",
    )
    """
    adata_sub = filter_adata(adata, filt_cells=filt_cells)
    logging.info(f"Filtered original adata of {adata.shape} to {adata_sub.shape}")

    if keyadded in adata_sub.uns:
        logging.warn(f"Key {keyadded} already present, removing")
        adata_sub.uns.pop(keyadded)
    sc.tl.rank_genes_groups(
        adata_sub, groupby=groupby, n_genes=n_genes, method=method, key_added=keyadded
    )
    return adata_sub


def flatten_marker_genes(marker_dict: dict, padj_cutoff: float = 0.05) -> List[str]:
    """
    Flatten the marker gene dict (i.e. adata.uns['rank_genes_groups'])
    Also makes sure that all returned genes are significant
    """
    # Just naively flattening the list using itertools.chain.from_iterable
    # may include genes that are not actually significant
    assert (
        "pvals_adj" in marker_dict and "names" in marker_dict
    ), f"Could not find requisite keys in marker dict with keys: {marker_dict.keys()}"

    # The lenth here is the number of genes given to find_marker_genes
    retval = set()
    assert len(marker_dict["pvals_adj"]) == len(marker_dict["names"])
    for i, (pvals, names) in enumerate(
        zip(marker_dict["pvals_adj"], marker_dict["names"])
    ):
        assert len(pvals) == len(names)  # Equivalent to number of clusters
        retval.update([n for n, p in zip(names, pvals) if p < padj_cutoff])
    return sorted(list(retval))


def attach_marker_features(adata: AnnData, ft: List[str]) -> None:
    """
    Given a list of "important" or high confidence features
    Attach this matrix to the given adata object as .obsm['X_confident']
    Useful for specifying the subset of genes used for clustering
    """
    markers = adata[:, ft].X
    if not isinstance(markers, np.ndarray):
        markers = markers.toarray()
    adata.obsm["X_confident"] = markers


def evaluate_pairwise_cluster_distance(
    adata: AnnData, stratify: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate pairwise distances between cells with given stratification
    """
    assert stratify in adata.obs.columns, f"Stratify key not found: {stratify}"
    cells_by_group = collections.defaultdict(list)
    cells_idx_by_group = collections.defaultdict(list)
    for i, (cellname, group_id) in enumerate(adata.obs[stratify].iteritems()):
        cells_by_group[group_id].append(cellname)
        cells_idx_by_group[group_id].append(i)

    distances = pd.DataFrame(
        index=sorted(list(cells_by_group.keys())),
        columns=sorted(list(cells_by_group.keys())),
    )
    sds = pd.DataFrame(
        index=sorted(list(cells_by_group.keys())),
        columns=sorted(list(cells_by_group.keys())),
    )

    for i, j in itertools.combinations_with_replacement(cells_idx_by_group.keys(), 2):
        idx_combos = itertools.product(cells_idx_by_group[i], cells_idx_by_group[j])
        d = sklearn.metrics.pairwise_distances(
            adata.X[cells_idx_by_group[i]],
            adata.X[cells_idx_by_group[j]],
            n_jobs=multiprocessing.cpu_count(),
        )
        mean = np.mean(d)
        sd = np.std(d)
        distances.loc[i, j] = mean
        distances.loc[j, i] = mean
        sds.loc[i, j] = sd
        sds.loc[j, i] = sd
    return distances, sds


def evaluate_pairwise_cell_distance(
    adata1: AnnData, adata2: AnnData, method: str = "euclidean"
) -> pd.DataFrame:
    """
    Evaluate pairwise distances between cells among two adata objects
    """
    distances = pd.DataFrame(
        data=sklearn.metrics.pairwise_distances(adata1.X, adata2.X, metric=method),
        index=adata1.obs_names,
        columns=adata2.obs_names,
    )
    return distances


def reindex_adata_vars(adata: AnnData, target_vars: List[str]) -> AnnData:
    """Reindexes the adata to match the given var_list, verbatim"""
    assert len(adata.var_names) == adata.n_vars
    if not utils.is_all_unique(adata.var_names):
        logging.warn("De-duping variable names before reindexing")
        adata.var_names_make_unique()
    assert utils.is_all_unique(target_vars), "Target vars are not all unique"
    intersected = set(adata.var_names).intersection(target_vars)
    logging.info(
        f"Overlap of {len(intersected)}/{adata.n_vars}, 0 vector will be filled in for {len(target_vars) - len(intersected)} 'missing' features"
    )
    vars_to_cols = dict(zip(adata.var_names, utils.ensure_arr(adata.X).T))
    assert (
        len(vars_to_cols) == adata.n_vars
    ), f"Size mismatch: {len(vars_to_cols)} {adata.n_vars}"

    default_null = np.zeros(adata.n_obs)
    mat = np.vstack(
        [
            vars_to_cols[v] if v in vars_to_cols else np.copy(default_null)
            for v in target_vars
        ]
    ).T
    target_shape = (adata.n_obs, len(target_vars))
    assert mat.shape == target_shape, f"Size mismatch: {mat.shape} {target_shape}"

    retval = AnnData(mat)
    retval.obs_names = adata.obs_names
    retval.var_names = target_vars
    return retval


def load_shareseq_data(tissue: str, dirname: str, mode: str = "RNA") -> AnnData:
    """Load the SHAREseq data"""
    assert os.path.isdir(dirname)
    atac_fname_dict = {
        "skin": [
            "GSM4156597_skin.late.anagen.barcodes.txt.gz",
            "GSM4156597_skin.late.anagen.counts.txt.gz",
            "GSM4156597_skin.late.anagen.peaks.bed.gz",
        ],
        "brain": [
            "GSM4156599_brain.barcodes.txt.gz",
            "GSM4156599_brain.counts.txt.gz",
            "GSM4156599_brain.peaks.bed.gz",
        ],
        "lung": [
            "GSM4156600_lung.barcodes.txt.gz",
            "GSM4156600_lung.counts.txt.gz",
            "GSM4156600_lung.peaks.bed.gz",
        ],
    }
    rna_fname_dict = {
        "skin": "GSM4156608_skin.late.anagen.rna.counts.txt.gz",
        "brain": "GSM4156610_brain.rna.counts.txt.gz",
        "lung": "GSM4156611_lung.rna.counts.txt.gz",
    }
    assert atac_fname_dict.keys() == rna_fname_dict.keys()
    assert tissue in atac_fname_dict.keys(), f"Unrecognized tissue: {tissue}"

    atac_barcodes_fname, atac_counts_fname, atac_peaks_fname = atac_fname_dict[tissue]
    assert "barcodes" in atac_barcodes_fname  # Check fnames are unpacked correctly
    assert "counts" in atac_counts_fname
    assert "peaks" in atac_peaks_fname
    atac_cell_barcodes = pd.read_csv(
        os.path.join(dirname, atac_barcodes_fname),
        delimiter="\t",
        index_col=0,
        header=None,
    )
    atac_cell_barcodes.index = [i.replace(",", ".") for i in atac_cell_barcodes.index]

    # Load in RNA data
    if mode == "RNA":
        retval = ad.read_text(os.path.join(dirname, rna_fname_dict[tissue])).T
        # Ensure that we return a sparse matrix as the underlying datatype
        retval.X = scipy.sparse.csr_matrix(retval.X)
        # Fix formatting of obs names where commas were used for periods
        retval.obs.index = [i.replace(",", ".") for i in retval.obs.index]
        intersected_barcodes = [
            bc for bc in retval.obs_names if bc in set(atac_cell_barcodes.index)
        ]
        assert intersected_barcodes, f"No common barcodes between RNA/ATAC for {tissue}"
        logging.info(
            f"RNA {tissue} intersects {len(intersected_barcodes)}/{len(retval.obs_names)} barcodes with ATAC"
        )
        retval = retval[intersected_barcodes]

    elif mode == "ATAC":
        # Load in ATAC data
        # read_mtx automatically gives us a sparse matrix
        retval = ad.read_mtx(os.path.join(dirname, atac_counts_fname)).T
        # Attach metadata
        retval.obs = atac_cell_barcodes
        atac_peaks = pd.read_csv(
            os.path.join(dirname, atac_peaks_fname),
            delimiter="\t",
            header=None,
            names=["chrom", "start", "end"],
        )
        atac_peaks.index = [f"{c}:{s}-{e}" for _i, c, s, e in atac_peaks.itertuples()]
        retval.var = atac_peaks
    else:
        raise ValueError("mode must be either RNA or ATAC")
    assert isinstance(retval.X, scipy.sparse.csr_matrix)
    return retval


def main():
    """On the fly debugging"""
    import glob

    merge_adata(
        [
            ad.read_h5ad(fname)
            for fname in glob.glob(
                "/home/wukevin/projects/commonspace_eval/test_multi/model*.h5ad"
            )
        ]
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
