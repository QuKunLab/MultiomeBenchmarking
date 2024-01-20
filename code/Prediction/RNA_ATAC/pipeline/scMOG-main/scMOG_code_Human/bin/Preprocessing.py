"""
Preprocessing
"""

import os
import sys
import logging
import argparse
import copy
import functools


SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scMOG"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

import sc_data_loaders
import adata_utils
import utils


logging.basicConfig(level=logging.INFO)


def build_parser():
    """Building a parameter parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data", "-d", type=str, nargs="*", help="Data files to train on",
    )
    input_group.add_argument(
        "--snareseq",
        action="store_true",
        help="Data in SNAREseq format, use custom data loading logic for separated RNA ATC files",
    )
    input_group.add_argument(
        "--shareseq",
        nargs="+",
        type=str,
        choices=["lung", "skin", "brain"],
        help="Load in the given SHAREseq datasets",
    )
    parser.add_argument(
        "--nofilter",
        action="store_true",
        help="Whether or not to perform filtering (only applies with --data argument)",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Do clustering data splitting in linear instead of log space",
    )
    parser.add_argument(
        "--clustermethod",
        type=str,
        choices=["leiden", "louvain"],
        default=None,
        help="Clustering method to determine data splits",
    )
    parser.add_argument(
        "--validcluster", type=int, default=0, help="Cluster ID to use as valid cluster"
    )
    parser.add_argument(
        "--testcluster", type=int, default=1, help="Cluster ID to use as test cluster"
    )
    parser.add_argument(
        "--outdir", "-o", required=True, type=str, help="Directory to output to"
    )
    parser.add_argument(
        "--ext",
        type=str,
        choices=["png", "pdf", "jpg"],
        default="pdf",
        help="Output format for plots",
    )
    return parser




def main():
    """Run Script"""
    parser = build_parser()
    args = parser.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # Specify output log file
    logger = logging.getLogger()
    fh = logging.FileHandler(f"{args.outdir}_training.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log parameters and pytorch version

    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    # Prepare RNA data
    logging.info("Reading RNA data")
    if args.snareseq:
        rna_data_kwargs = copy.copy(sc_data_loaders.SNARESEQ_RNA_DATA_KWARGS)
        
    elif args.shareseq:
        logging.info(f"Loading in SHAREseq RNA data for: {args.shareseq}")
        rna_data_kwargs = copy.copy(sc_data_loaders.SNARESEQ_RNA_DATA_KWARGS)
        rna_data_kwargs["fname"] = None
        rna_data_kwargs["reader"] = None
        rna_data_kwargs["cell_info"] = None
        rna_data_kwargs["gene_info"] = None
        rna_data_kwargs["transpose"] = False
        # Load in the datasets
        shareseq_rna_adatas = []
        for tissuetype in args.shareseq:
            shareseq_rna_adatas.append(
                adata_utils.load_shareseq_data(
                    tissuetype,
                    dirname="/data/GSE140203_SHAREseq",
                    mode="RNA",
                )
            )
        shareseq_rna_adata = shareseq_rna_adatas[0]
        if len(shareseq_rna_adatas) > 1:
            shareseq_rna_adata = shareseq_rna_adata.concatenate(
                *shareseq_rna_adatas[1:],
                join="inner",
                batch_key="tissue",
                batch_categories=args.shareseq,
            )
        rna_data_kwargs["raw_adata"] = shareseq_rna_adata
    else:
        rna_data_kwargs = copy.copy(sc_data_loaders.TENX_PBMC_RNA_DATA_KWARGS)
        rna_data_kwargs["fname"] = args.data
        if args.nofilter:
            rna_data_kwargs = {
                k: v for k, v in rna_data_kwargs.items() if not k.startswith("filt_")
            }
    ##The following is the training set and test set etc. divided according to clusters
    rna_data_kwargs["data_split_by_cluster_log"] = not args.linear
    rna_data_kwargs["data_split_by_cluster"] = args.clustermethod

    sc_rna_dataset = sc_data_loaders.SingleCellDataset(
        valid_cluster_id=args.validcluster,
        test_cluster_id=args.testcluster,
        **rna_data_kwargs,
    )

    sc_rna_train_dataset = sc_data_loaders.SingleCellDatasetSplit(
        sc_rna_dataset, split="train",
    )
    sc_rna_valid_dataset = sc_data_loaders.SingleCellDatasetSplit(
        sc_rna_dataset, split="valid",
    )
    sc_rna_test_dataset = sc_data_loaders.SingleCellDatasetSplit(
        sc_rna_dataset, split="test",
    )

    # Prepare ATAC data
    logging.info("Aggregating ATAC clusters")
    if args.snareseq:
        atac_data_kwargs = copy.copy(sc_data_loaders.SNARESEQ_ATAC_DATA_KWARGS)
    elif args.shareseq:
        logging.info(f"Loading in SHAREseq ATAC data for {args.shareseq}")
        atac_data_kwargs = copy.copy(sc_data_loaders.SNARESEQ_ATAC_DATA_KWARGS)
        atac_data_kwargs["reader"] = None
        atac_data_kwargs["fname"] = None
        atac_data_kwargs["cell_info"] = None
        atac_data_kwargs["gene_info"] = None
        atac_data_kwargs["transpose"] = False
        atac_adatas = []
        for tissuetype in args.shareseq:
            atac_adatas.append(
                adata_utils.load_shareseq_data(
                    tissuetype,
                    dirname="GSE140203_SHAREseq",
                    mode="ATAC",
                )
            )
        atac_bins = [a.var_names for a in atac_adatas]
        if len(atac_adatas) > 1:
            atac_bins_harmonized = sc_data_loaders.harmonize_atac_intervals(*atac_bins)
            atac_adatas = [
                sc_data_loaders.repool_atac_bins(a, atac_bins_harmonized)
                for a in atac_adatas
            ]
        shareseq_atac_adata = atac_adatas[0]
        if len(atac_adatas) > 1:
            shareseq_atac_adata = shareseq_atac_adata.concatenate(
                *atac_adatas[1:],
                join="inner",
                batch_key="tissue",
                batch_categories=args.shareseq,
            )
        atac_data_kwargs["raw_adata"] = shareseq_atac_adata
    else:
        atac_parsed = [
            utils.sc_read_10x_h5_ft_type(fname, "Peaks") for fname in args.data
        ]
        if len(atac_parsed) > 1:
            atac_bins = sc_data_loaders.harmonize_atac_intervals(
                atac_parsed[0].var_names, atac_parsed[1].var_names
            )
            for bins in atac_parsed[2:]:
                atac_bins = sc_data_loaders.harmonize_atac_intervals(
                    atac_bins, bins.var_names
                )
            logging.info(f"Aggregated {len(atac_bins)} bins")
        else:
            atac_bins = list(atac_parsed[0].var_names)

        atac_data_kwargs = copy.copy(sc_data_loaders.TENX_PBMC_ATAC_DATA_KWARGS)
        atac_data_kwargs["fname"] = rna_data_kwargs["fname"]
        atac_data_kwargs["pool_genomic_interval"] = 0  # Do not pool
        atac_data_kwargs["reader"] = functools.partial(
            utils.sc_read_multi_files,
            reader=lambda x: sc_data_loaders.repool_atac_bins(
                utils.sc_read_10x_h5_ft_type(x, "Peaks"), atac_bins,
            ),
        )
    atac_data_kwargs["cluster_res"] = 0  # Do not bother clustering ATAC data

    sc_atac_dataset = sc_data_loaders.SingleCellDataset(
        predefined_split=sc_rna_dataset, **atac_data_kwargs
    )
    sc_atac_train_dataset = sc_data_loaders.SingleCellDatasetSplit(
        sc_atac_dataset, split="train",
    )
    sc_atac_valid_dataset = sc_data_loaders.SingleCellDatasetSplit(
        sc_atac_dataset, split="valid",
    )
    sc_atac_test_dataset = sc_data_loaders.SingleCellDatasetSplit(
        sc_atac_dataset, split="test",
    )
    outdir_name = (args.outdir)
    if not os.path.isdir(outdir_name):
        assert not os.path.exists(outdir_name)
        os.makedirs(outdir_name)
    assert os.path.isdir(outdir_name)
    with open(os.path.join(outdir_name, "rna_genes.txt"), "w") as sink:
        for gene in sc_rna_dataset.data_raw.var_names:
            sink.write(gene + "\n")
    with open(os.path.join(outdir_name, "atac_bins.txt"), "w") as sink:
        for atac_bin in sc_atac_dataset.data_raw.var_names:
            sink.write(atac_bin + "\n")

    # Write dataset
    ### Full
    sc_rna_dataset.size_norm_counts.write_h5ad(
        os.path.join(outdir_name, "full_rna.h5ad")
    )
    sc_rna_dataset.size_norm_log_counts.write_h5ad(
        os.path.join(outdir_name, "full_rna_log.h5ad")
    )
    sc_atac_dataset.data_raw.write_h5ad(os.path.join(outdir_name, "full_atac.h5ad"))
    ### Train
    sc_rna_train_dataset.size_norm_counts.write_h5ad(
        os.path.join(outdir_name, "train_rna.h5ad")
    )
    sc_atac_train_dataset.data_raw.write_h5ad(
        os.path.join(outdir_name, "train_atac.h5ad")
    )
    ### Valid
    sc_rna_valid_dataset.size_norm_counts.write_h5ad(
        os.path.join(outdir_name, "valid_rna.h5ad")
    )
    sc_atac_valid_dataset.data_raw.write_h5ad(
        os.path.join(outdir_name, "valid_atac.h5ad")
    )
    ### Test
    sc_rna_test_dataset.size_norm_counts.write_h5ad(
        os.path.join(outdir_name, "truth_rna.h5ad")
    )
    sc_atac_test_dataset.data_raw.write_h5ad(
        os.path.join(outdir_name, "truth_atac.h5ad")
    )




if __name__ == "__main__":
    main()
