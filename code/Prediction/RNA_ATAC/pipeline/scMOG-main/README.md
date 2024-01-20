# scMOG

Recent advances in single-cell sequencing technology have made it possible to measure multiple paired omics simultaneously in a single cell such as CITE-seq and SNARE-seq. Yet the widespread application of those single-cell multiomics profiling technologies has been limited by their experimental complexity, noise in nature, and high cost. In addition, single-omics sequencing technologies have generated tremendous and high-quality single-cell datasets but have yet to be fully utilized. Here, we develop scMOG, a deep learning-based framework to in silico generate single-cell ATAC data from experimentally available single-cell RNA-seq measurements and vice versa. 

![图片1](https://github.com/lanlemon/111/assets/68333653/1a4e9638-cafc-40ac-937c-31af7236bd0b)

For more information, please see to our paper:
[Efficient generation of paired single-cell multiomics profiles by deep learning](https://onlinelibrary.wiley.com/doi/10.1002/advs.202301169)

## Installation

To install scMOG, make sure you have [PyTorch](https://pytorch.org/) and [scanpy](https://scanpy.readthedocs.io/en/stable/) installed. If you need more details on the dependences, look at the `environment.yml` file. Set up conda environment for scMOG：
```
    conda env create -f environment.yml
```

## Preprocessing Datasets
scMOG is trained using paired scRNA-seq/scATAC-seq measurements. We provide the data pre-processing code. The user only needs to input the 'h5' file of the data to perform the corresponding data pre-processing（Each h5 file must contain both RNA and ATAC paired modalities）：

```bash
python bin/Preprocessing.py --data FILE1.h5 FILE2.h5 --outdir Preprocessed_datasets
```
In addition, we also support other multi-omics dataset inputs such as SHARE-seq, SNARE-seq, for example:
```bash
python bin/Preprocessing.py --snareseq --outdir snareseq_datasets
```

### Train model
With the datasets obtained after pre-processing, we can then train the model:

```bash
python bin/train.py --outdir output
```
This training script will create a new directory output that contains:
* `***.pth` files, which contain the trained model parameters.
* `loss.pdf` files, which is used as an evaluation indicator of the model, which reflects the degree of fit between the predicted and true values of the model
*  `*.pdf` files that contain summary test set metrics such as correlation , AUPRC and AUROC.

### Generation on other datasets
Once trained, scMOG can generate paired datasets from other datasets using the following example command.

```bash
python bin/predict-rna.py --outdir Otherdataset_generation 
python bin/predict-atac.py --outdir Otherdataset_generation 
```
 scMOG will create its outputs in the folder `Otherdataset_generation` accordingly:

* Various `*.h5ad` files containing the predictions. These are named with the convention `inputMode_outputMode_adata.h5ad`. For example the file `atac_rna_adata.h5ad` contains the RNA predictions from ATAC input.
* If given paired data, this script will also generate concordance metrics in `*.pdf` files with a similar naming convention. For example, `atac_rna_log.pdf` will contain a log-scaled scatterplot comparing measured and imputed expression values per gene per cell.


### Additional commandline options
All the above scripts have some options designed for advanced users, exposing some features such as clustering methods, learning rates, etc. Users can adjust them by themselves.


