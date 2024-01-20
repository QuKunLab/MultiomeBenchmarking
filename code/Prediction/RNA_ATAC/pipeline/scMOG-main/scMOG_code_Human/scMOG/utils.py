"""
Utility functions

Some functions live here because otherwise managing their import
in other places would be overly difficult
"""
import os
import sys
import functools
import logging
from typing import *
import itertools
import collections
import gzip

import numpy as np
import pandas as pd
import scipy
import scanpy as sc
from anndata import AnnData

import torch

import intervaltree as itree
import sortedcontainers

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
assert os.path.isdir(DATA_DIR)
HG38_GTF = os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.100.gtf.gz")
assert os.path.isfile(HG38_GTF)
HG19_GTF = os.path.join(DATA_DIR, "Homo_sapiens.GRCh37.87.gtf.gz")
assert os.path.isfile(HG19_GTF)


def ensure_arr(x) -> np.ndarray:
    """Return x as a np.array"""
    if isinstance(x, np.matrix):
        return np.squeeze(np.asarray(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        return x.toarray()
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    else:
        raise TypeError(f"Unrecognized type: {type(x)}")


def is_integral_val(x) -> bool:
    """
    Check if value(s) can be cast as integer without losing precision
    >>> is_integral_val(np.array([1., 2., 3.]))
    True
    >>> is_integral_val(np.array([1., 2., 3.5]))
    False
    """
    if isinstance(x, (np.ndarray, scipy.sparse.csr_matrix)):
        x_int = x.astype(int)
    else:
        x_int = int(x)
    residuals = x - x_int
    if isinstance(residuals, scipy.sparse.csr_matrix):
        residuals = ensure_arr(residuals[residuals.nonzero()])
    return np.all(np.isclose(residuals, 0))


def get_file_extension_no_gz(fname: str) -> str:
    """
    Get the filename extension (not gz)
    >>> get_file_extension_no_gz("foo.txt.gz")
    'txt'
    >>> get_file_extension_no_gz("foo.bar")
    'bar'
    >>> get_file_extension_no_gz("foo")
    ''
    """
    assert fname, f"Got empty input"
    retval = ""
    while fname and (not retval or retval == ".gz"):
        fname, ext = os.path.splitext(fname)
        if not ext:
            break  # Returns empty string
        if ext != ".gz":
            retval = ext
    return retval.strip(".")


def get_ad_reader(fname: str, ft_type: str) -> Callable:
    """Return the function that when called, returns an AnnData object"""
    # Modality is only used for reading h5 files
    ext = get_file_extension_no_gz(fname)
    if ext == "h5":
        pfunc = functools.partial(sc.read_10x_h5, gex_only=False)
        if not ft_type:
            return pfunc

        def helper(fname, pfunc, ft_type):
            a = pfunc(fname)
            return a[:, a.var["feature_types"] == ft_type]

        return functools.partial(helper, pfunc=pfunc, ft_type=ft_type)
    elif ext == "h5ad":
        return sc.read_h5ad
    elif ext == "csv":
        return sc.read_csv
    elif ext in ("tsv", "txt"):
        return sc.read_text
    else:
        raise ValueError("Could not determine reader for {fname}")


def sc_read_multi_files(
    fnames: List[str],
    reader: Callable = None,
    feature_type: str = "",
    transpose: bool = False,
    var_name_sanitization: Callable = None,
    join: str = "inner",
) -> AnnData:
    """Given a list of files, read the adata objects and concatenate"""
    # var name sanitization lets us make sure that variable name conventions are consistent
    assert fnames
    for fname in fnames:
        assert os.path.isfile(fname), f"File does not exist: {fname}"
    if reader is None:  # Autodetermine reader type
        parsed = [get_ad_reader(fname, feature_type)(fname) for fname in fnames]
    else:  # Given a fixed reader
        parsed = [reader(fname) for fname in fnames]
    if transpose:
        # h5 reading automatically transposes
        parsed = [
            p.T if get_file_extension_no_gz(fname) != "h5" else p
            for p, fname in zip(parsed, fnames)
        ]

    # Log and check genomes
    for f, p in zip(fnames, parsed):
        logging.info(f"Read in {f} for {p.shape} (obs x var)")
    genomes_present = set(
        g
        for g in itertools.chain.from_iterable(
            [p.var["genome"] for p in parsed if "genome" in p.var]
        )
        if g
    )

    # Build concatenated output
    assert len(genomes_present) <= 1, f"Got more than one genome: {genomes_present}"
    for fname, p in zip(fnames, parsed):  # Make variable names unique and ensure sparse
        if var_name_sanitization:
            p.var.index = pd.Index([var_name_sanitization(i) for i in p.var_names])
        p.var_names_make_unique()
        p.X = scipy.sparse.csr_matrix(p.X)
        p.obs["source_file"] = fname
    retval = parsed[0]
    if len(parsed) > 1:
        retval = retval.concatenate(*parsed[1:], join=join)
    return retval


def sc_read_10x_h5_ft_type(fname: str, ft_type: str) -> AnnData:
    """Read the h5 file, taking only features with specified ft_type"""
    assert fname.endswith(".h5")
    parsed = sc.read_10x_h5(fname, gex_only=False)
    parsed.var_names_make_unique()
    assert ft_type in set(
        parsed.var["feature_types"]
    ), f"Given feature type {ft_type} not in included types: {set(parsed.var['feature_types'])}"

    retval = parsed[
        :,
        [n for n in parsed.var_names if parsed.var.loc[n, "feature_types"] == ft_type],
    ]
    return retval


def extract_file(fname, overwrite: bool = False) -> str:
    """Extracts the file and return the path to extracted file"""
    out_fname = os.path.abspath(fname.replace(".gz", ""))
    if os.path.isfile(out_fname):  # If the file already
        # If the file already exists and we aren't overwriting, do nothing
        if not overwrite:
            return out_fname
        os.remove(out_fname)
    with open(out_fname, "wb") as sink, gzip.GzipFile(fname) as source:
        sink.write(source.read())
    return out_fname


def read_delimited_file(
    fname: str, delimiter: str = "\n", comment: str = "#"
) -> List[str]:
    """Read the delimited (typically newline) file into a list"""
    with open(fname) as source:
        contents = source.read().strip()
        retval = contents.split(delimiter)
    # Filter out comment entries
    retval = [item for item in retval if not item.startswith(comment)]
    return retval


@functools.lru_cache(maxsize=2, typed=True)
def read_gtf_gene_to_pos(
    fname: str = HG38_GTF,
    acceptable_types: List[str] = None,
    addtl_attr_filters: dict = None,
    extend_upstream: int = 0,
    extend_downstream: int = 0,
) -> Dict[str, Tuple[str, int, int]]:
    """
    Given a gtf file, read it in and return as a ordered dictionary mapping genes to genomic ranges
    Ordering is done by chromosome then by position
    """
    # https://uswest.ensembl.org/info/website/upload/gff.html
    gene_to_positions = collections.defaultdict(list)
    gene_to_chroms = collections.defaultdict(set)

    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            if line.startswith(b"#"):
                continue
            line = line.decode()
            (
                chrom,
                entry_type,
                entry_class,
                start,
                end,
                score,
                strand,
                frame,
                attrs,
            ) = line.strip().split("\t")
            assert strand in ("+", "-")
            if acceptable_types and entry_type not in acceptable_types:
                continue
            attr_dict = dict(
                [t.strip().split(" ", 1) for t in attrs.strip().split(";") if t]
            )
            if addtl_attr_filters:
                tripped_attr_filter = False
                for k, v in addtl_attr_filters.items():
                    if k in attr_dict:
                        if isinstance(v, str):
                            if v != attr_dict[k].strip('"'):
                                tripped_attr_filter = True
                                break
                        else:
                            raise NotImplementedError
                if tripped_attr_filter:
                    continue
            gene = attr_dict["gene_name"].strip('"')
            start = int(start)
            end = int(end)
            assert (
                start <= end
            ), f"Start {start} is not less than end {end} for {gene} with strand {strand}"
            if extend_upstream:
                if strand == "+":
                    start -= extend_upstream
                else:
                    end += extend_upstream
            if extend_downstream:
                if strand == "+":
                    end += extend_downstream
                else:
                    start -= extend_downstream

            gene_to_positions[gene].append(start)
            gene_to_positions[gene].append(end)
            gene_to_chroms[gene].add(chrom)

    slist = sortedcontainers.SortedList()
    for gene, chroms in gene_to_chroms.items():
        if len(chroms) != 1:
            logging.warn(
                f"Got multiple chromosomes for gene {gene}: {chroms}, skipping"
            )
            continue
        positions = gene_to_positions[gene]
        t = (chroms.pop(), min(positions), max(positions), gene)
        slist.add(t)

    retval = collections.OrderedDict()
    for chrom, start, stop, gene in slist:
        retval[gene] = (chrom, start, stop)
    return retval


@functools.lru_cache(maxsize=2)
def read_gtf_gene_symbol_to_id(
    fname: str = HG38_GTF,
    acceptable_types: List[str] = None,
    addtl_attr_filters: dict = None,
) -> Dict[str, str]:
    """Return a map from easily readable gene name to ENSG gene ID"""
    retval = {}
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            if line.startswith(b"#"):
                continue
            line = line.decode()
            (
                chrom,
                entry_type,
                entry_class,
                start,
                end,
                score,
                strand,
                frame,
                attrs,
            ) = line.strip().split("\t")
            assert strand in ("+", "-")
            if acceptable_types and entry_type not in acceptable_types:
                continue
            attr_dict = dict(
                [t.strip().split(" ", 1) for t in attrs.strip().split(";") if t]
            )
            if addtl_attr_filters:
                tripped_attr_filter = False
                for k, v in addtl_attr_filters.items():
                    if k in attr_dict:
                        if isinstance(v, str):
                            if v != attr_dict[k].strip('"'):
                                tripped_attr_filter = True
                                break
                        else:
                            raise NotImplementedError
                if tripped_attr_filter:
                    continue
            gene = attr_dict["gene_name"].strip('"')
            gene_id = attr_dict["gene_id"].strip('"')
            retval[gene] = gene_id

    return retval


@functools.lru_cache(maxsize=8)
def read_gtf_pos_to_features(
    fname: str = HG38_GTF,
    acceptable_types=[],
    addtl_attr_filters: dict = None,
) -> Dict[str, itree.IntervalTree]:
    """Return an intervaltree representation of the gtf file"""
    acceptable_types = set(acceptable_types)
    retval = collections.defaultdict(itree.IntervalTree)
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            line = line.decode()
            if line.startswith("#"):
                continue
            (
                chrom,
                entry_type,
                entry_class,
                start,
                end,
                score,
                strand,
                frame,
                attrs,
            ) = line.strip().split("\t")
            start = int(start)
            end = int(end)
            if start >= end:
                continue
            assert strand in ("+", "-")
            if acceptable_types and entry_type not in acceptable_types:
                continue
            attr_dict = dict(
                [
                    [u.strip('"') for u in t.strip().split(" ", 1)]
                    for t in attrs.strip().split(";")
                    if t
                ]
            )
            if addtl_attr_filters:
                tripped_attr_filter = False
                for k, v in addtl_attr_filters.items():
                    if k in attr_dict:
                        if isinstance(v, str):
                            if v != attr_dict[k].strip('"'):
                                tripped_attr_filter = True
                                break
                        else:
                            raise NotImplementedError
                if tripped_attr_filter:
                    continue
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            assert (
                "entry_type" not in attr_dict
                and "entry_class" not in attr_dict
                and "entry_strand" not in attr_dict
            )
            attr_dict["entry_type"] = entry_type
            attr_dict["entry_class"] = entry_class
            attr_dict["entry_strand"] = strand
            retval[chrom][int(start) : int(end)] = attr_dict
    return retval


def isnotebook() -> bool:
    """
    Returns True if the current execution environment is a jupyter notebook
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_device(i: int = None) -> str:
    """Returns the i-th GPU if GPU is available, else CPU"""
    if torch.cuda.is_available() and isinstance(i, int):
        devices = list(range(torch.cuda.device_count()))
        device_idx = devices[i]
        torch.cuda.set_device(device_idx)
        d = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(d)
    else:
        d = torch.device("cpu")
    return d


def is_numeric(x) -> bool:
    """Return True if x is numeric"""
    try:
        x = float(x)
        return True
    except ValueError:
        return False


def is_all_unique(x: Iterable[Any]) -> bool:
    """
    Return whether the given iterable is all unique
    >>> is_all_unique(['x', 'y'])
    True
    >>> is_all_unique(['x', 'x', 'y'])
    False
    """
    return len(set(x)) == len(x)


def shifted_sigmoid(x, center: float = 0.5, slope: float = 25):
    """Compute a shifted sigmoid with configurable center and slope (steepness)"""
    return 1.0 / (1.0 + np.exp(slope * (-x + center)))


def unit_rescale(vals):
    """Rescale the given values to be between 0 and 1"""
    vals = np.array(vals).astype(float)
    denom = float(np.max(vals) - np.min(vals))
    retval = (vals - np.min(vals)) / denom
    assert np.alltrue(retval <= 1.0) and np.alltrue(retval >= 0.0)
    return retval


def split_df_by_col(df: pd.DataFrame, col: str) -> List[pd.DataFrame]:
    """Splits the dataframe into multiple dataframes by value of col"""
    unique_vals = set(df[col])

    retval = {}
    for v in unique_vals:
        df_sub = df[df[col] == v]
        retval[v] = df_sub
    return retval


def main():
    """On the fly testing"""
    x = read_gtf_pos_to_features(acceptable_types=["havana"])
    # print(x)


if __name__ == "__main__":
    get_file_extension_no_gz("foo")
    import doctest

    doctest.testmod()
    main()
