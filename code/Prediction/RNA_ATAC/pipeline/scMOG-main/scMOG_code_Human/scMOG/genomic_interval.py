"""
A GenomicInterval object specific a genomic interval and allows us to
perform easy queries between different intervals
"""

import os
import sys
from typing import *

import utils

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
assert os.path.isdir(DATA_DIR)
MM9_GTF = os.path.join(DATA_DIR, "Mus_musculus.NCBIM37.67.gtf.gz")
assert os.path.isfile(MM9_GTF)
HG38_GTF = os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.100.gtf.gz")
assert os.path.isfile(HG38_GTF)
HG19_GTF = os.path.join(DATA_DIR, "Homo_sapiens.GRCh37.87.gtf.gz")
assert os.path.isfile(HG19_GTF)


class GenomicInterval(object):
    """
    Class that indicates a genomic interval
    """

    def __init__(
        self, interval: Union[str, Tuple[str, int, int]], metadata_dict: Dict = None
    ):
        """interval should follow the format chrX:123-456"""
        if isinstance(interval, str):
            if ":" in interval:
                chrom, span = interval.split(":")
                start, stop = span.split("-")
            else:  # Seurat style format
                chrom, start, stop = interval.split("-")
        else:
            chrom, start, stop = interval

        self.chrom = chrom if chrom.startswith("chr") else "chr" + chrom
        self.start = int(start)
        self.stop = int(stop)
        assert self.start < self.stop
        self.metadata = metadata_dict

    def __str__(self):
        return f"{self.chrom}:{self.start}-{self.stop}"

    def __lt__(self, other):
        if self.chrom != other.chrom:
            raise ValueError("Cannot compare intervals on different chromosomes")
        return self.start < other.start

    def __gt__(self, other):
        """
        >>> x = GenomicInterval("chr1:100-200")
        >>> y = GenomicInterval("chr1:10-20")
        >>> x > y
        True
        """
        if self.chrom != other.chrom:
            raise ValueError("Cannot compare intervals on different chromosomes")
        return self.start > other.start

    def __eq__(self, other):
        return (
            self.chrom == other.chrom
            and self.start == other.start
            and self.stop == other.stop
        )

    def as_tuple(self) -> Tuple[str, int, int]:
        """Returns the representation of genomic interval as a tuple"""
        return (self.chrom, self.start, self.stop)

    def contains(self, query) -> bool:
        """
        Returns whether the query is fully contained in self
        >>> x = GenomicInterval("chr1:1000-2000")
        >>> y = GenomicInterval("chr1:999-2000")
        >>> x.contains(y)
        False
        >>> y.contains(x)
        True
        """
        if query.chrom != self.chrom:
            return False

        return self.start <= query.start and self.stop >= query.stop

    def overlaps(self, query) -> bool:
        """
        Returns whether the two overlap

        >>> x = GenomicInterval("chr1:1000-2000")
        >>> y = GenomicInterval("chr1:999-2000")
        >>> x.overlaps(y)
        True
        >>> y.overlaps(x)
        True
        >>> z = GenomicInterval("chr1:1-1000")
        >>> z.overlaps(x)
        True
        >>> a = GenomicInterval("chr1:1500-3000")
        >>> a.overlaps(x)
        True
        >>> x.overlaps(a)
        True
        """
        if isinstance(query, str):
            query = GenomicInterval(query)

        if query.chrom != self.chrom:
            return False
        if self.contains(query) or query.contains(self) or self == query:
            return True
        if (self.start <= query.start <= self.stop) or (
            self.start <= query.stop <= self.stop
        ):
            return True
        return False

    def expand(self, size: int, fiveprime: bool = True, threeprime: bool = True):
        """Expand this current interval by a certain amount"""
        assert fiveprime or threeprime
        if fiveprime:
            self.start -= size
        if threeprime:
            self.stop += size

    def difference(self, other) -> int:
        """
        Return the difference between two intervals
        >>> x = GenomicInterval("chr1:100-200")
        >>> y = GenomicInterval("chr1:250-300")
        >>> x.difference(y)
        -50
        >>> y.difference(x)
        50
        """
        assert self.chrom == other.chrom
        if self.contains(other) or other.contains(self) or self.overlaps(other):
            return 0
        # Calculate an actual difference
        if self < other:
            return self.stop - other.start
        else:
            return self.start - other.stop

    @property
    def size(self) -> int:
        """
        Return the size of this intervals span
        >>> x = GenomicInterval("chr1:100-200")
        >>> x.size
        100
        """
        return self.stop - self.start


def query_overlaps(query: str, intervals: List[str]) -> List[str]:
    """
    Given a query, find the intervals that overlap with it
    >>> query_overlaps("chr1:100-200", ["chr1:1-101", "chr2:100-200"])
    ['chr1:1-101']
    """
    query_gi = GenomicInterval(query)
    target_gis = [GenomicInterval(i) for i in intervals]

    target_gis = [
        str(gi)
        for gi in target_gis
        if gi.chrom == query_gi.chrom and gi.overlaps(query_gi)
    ]
    return target_gis


def from_gene(gene: str, reference_gtf: str = HG38_GTF) -> GenomicInterval:
    """Construct a genomic interval from the gene"""
    gtf_parsed = utils.read_gtf_gene_to_pos(reference_gtf)
    interval_str = gtf_parsed[gene]
    return GenomicInterval(interval_str)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
