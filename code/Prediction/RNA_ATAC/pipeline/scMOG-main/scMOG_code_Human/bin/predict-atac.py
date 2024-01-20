"""
Code to predict atac
"""

import os
import sys
import logging
import argparse
import scanpy as sc
import itertools


import numpy as np
import pandas as pd
import scipy.spatial
import collections
import matplotlib.pyplot as plt

import torch
import torch.nn as nn



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scMOG"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

MODELS_DIR = os.path.join(SRC_DIR, "models")
assert os.path.isdir(MODELS_DIR)
sys.path.append(MODELS_DIR)


import sc_data_loaders
import adata_utils
import anndata as ad
import loss_functions
import activations
import plot_utils
import utils
import sklearn.metrics as metrics

import both_GAN_1

logging.basicConfig(level=logging.INFO)



SAVEFIG_DPI = 1200

def build_parser():
    """Building a parameter parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--outdir", "-o", required=True, type=str, help="Directory to output to"
    )
    parser.add_argument(
        "--hidden", type=int, nargs="*", default=[16], help="Hidden dimensions"
    )

    parser.add_argument(
        "--lr", "-l", type=float, default=[0.0001], nargs="*", help="Learning rate"
    )
    parser.add_argument(
        "--batchsize", "-b", type=int, nargs="*", default=[512], help="Batch size"
    )
    parser.add_argument(
        "--seed", type=int, nargs="*", default=[182822], help="Random seed to use"
    )
    parser.add_argument("--device", default=0, type=int, help="Device to train on")
    parser.add_argument(
        "--ext",
        type=str,
        choices=["png", "pdf", "jpg"],
        default="pdf",
        help="Output format for plots",
    )
    #parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    #parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    return parser



def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    Convert scipy's sparse matrix to torch's sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

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
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        raise TypeError(f"Unrecognized type: {type(x)}")

def plot_auroc(
        truth,
        preds,
        title_prefix: str = "Receiver operating characteristic",
        fname: str = "",
):
    """
    Plot AUROC after flattening inputs
    """
    truth = ensure_arr(truth).flatten()
    preds = ensure_arr(preds).flatten()
    # truth = truth.flatten()
    # preds = preds.flatten()
    fpr, tpr, _thresholds = metrics.roc_curve(truth, preds, pos_label=2)

    maxindex = (tpr - fpr).tolist().index(max(tpr- fpr))
    threshold = _thresholds[maxindex]
    print(_thresholds.shape)
    logging.info(f"_thresholds:{_thresholds[_thresholds.shape[0]//2]}")
    logging.info(f"threshold :{threshold}")

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

def plot_prc(
        truth,
        preds,
        title_prefix: str = "Receiver operating characteristic",
        fname: str = "",
):
    """
    Plot PRC after flattening inputs
    """
    truth = truth.flatten()
    preds = preds.flatten()
    precision, recall, _thresholds = metrics.precision_recall_curve(truth, preds)
    auc = metrics.auc(recall,precision)
    logging.info(f"Found AUPRC of {auc:.4f}")

    fig, ax = plt.subplots(dpi=300, figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set(
        xlim=(0, 1.0),
        ylim=(0.0, 1.05),
        xlabel="recall",
        ylabel="precision",
        title=f"{title_prefix} (PRC={auc:.2f})",
    )
    if fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
    return fig


def rmse_value(truth,
               preds,
):
    'Calculate RMSE'
    truth = truth.flatten()
    preds = preds.flatten()
    rmse=np.sqrt(metrics.mean_squared_error(truth, preds))
    logging.info(f"Found RMSE of {rmse:.4f}")

def auroc(
        truth,
        preds,

):
    """
    Plot AUROC after flattening inputs
    """
    auc=[]

    print(len(truth))
    for i in range(len(truth)):
        fpr, tpr, _thresholds = metrics.roc_curve(truth[i],preds[i])
        auc.append(metrics.auc(fpr, tpr))
    print(len(auc))


    return auc

def main():
    """Run Script"""
    parser = build_parser()
    args = parser.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # Specify output log file
    logger = logging.getLogger()
    fh = logging.FileHandler(f"{args.outdir}.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log parameters and pytorch version
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")

    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")



    sc_rna_dataset=ad.read_h5ad('truth_rna_GM.h5ad')
    sc_atac_dataset=ad.read_h5ad('truth_atac_GM.h5ad')

    cuda = True if torch.cuda.is_available() else False
# Model
    param_combos = list(
        itertools.product(
            args.hidden, args.lr, args.seed
        )
    )
    for h_dim, lr, rand_seed in param_combos:
        outdir_name = (
            f"{args.outdir}_hidden_{h_dim}_lr_{lr}_seed_{rand_seed}"
            if len(param_combos) > 1
            else args.outdir
        )
        if not os.path.isdir(outdir_name):
            assert not os.path.exists(outdir_name)
            os.makedirs(outdir_name)
        assert os.path.isdir(outdir_name)

        generator = both_GAN_1.GeneratorATAC(hidden_dim=h_dim,
                                                 input_dim1=sc_rna_dataset.X.shape[1],
                                                 input_dim2=sc_atac_dataset.X.shape[1],
                                                 # out_dim=get_per_chrom_feature_count(sc_atac_dataset),
                                                 final_activations2=nn.Sigmoid(),
                                                 flat_mode=True,
                                                 seed=rand_seed,
                                                 )

        # if cuda:
        #    generator.cuda()


        #RNA——》ATAC

        logging.info("Evaluating RNA > ATAC")
        sc_rna_test_dataset = ad.read_h5ad('truth_rna_GM.h5ad')
        Acoo = sc_rna_test_dataset.X.tocoo()
        sc_rna_test = scipy_sparse_mat_to_torch_sparse_tensor(Acoo)
        sc_rna_test = sc_rna_test.to_dense()


        sc_atac_test_dataset = ad.read_h5ad('truth_atac_GM.h5ad')
        Acoo1 = sc_atac_test_dataset.X.tocoo()
        sc_atac_test= scipy_sparse_mat_to_torch_sparse_tensor(Acoo1)
        sc_atac_test = sc_atac_test.to_dense()



        test_iter = torch.utils.data.DataLoader(dataset=sc_rna_test, batch_size=64 )
        def pridect(test_iter):
           generator.eval()
           first=1
           generator.load_state_dict(torch.load('ATACgenerator.pth',map_location='cpu'))
           for x in test_iter:
              # if cuda:
              #      x = x.cuda()
              with torch.no_grad():
                  y_pred = generator(x)
                  if first==1:
                      ret=y_pred
                      first=0
                  else:
                      ret=torch.cat((ret,y_pred),0)
           return ret

        sc_rna_atac_test_preds =pridect(test_iter)



        sc_rna_atac_full_preds_anndata = sc.AnnData(
            scipy.sparse.csr_matrix(sc_rna_atac_test_preds),
            obs=sc_rna_test_dataset.obs,
        )
        sc_rna_atac_full_preds_anndata.var_names =sc_atac_dataset.var_names
        logging.info("Writing ATAC from RNA")

        # Seurat also expects the raw attribute to be populated
        #sc_atac_rna_full_preds_anndata.raw = sc_atac_rna_full_preds_anndata.copy()
        sc_rna_atac_full_preds_anndata.write(
            os.path.join(args.outdir, f"rna_atac_adata_final.h5ad"))



        fig = plot_auroc(
            sc_atac_test,
            sc_rna_atac_test_preds,
            title_prefix="RNA > ATAC",
            fname=os.path.join(outdir_name, f"rna_atac_auroc.{args.ext}"),
        )
        plt.close(fig)

        pbmc_atac2 = ad.read_h5ad(os.path.join(args.outdir, f"rna_atac_adata_final.h5ad"))
        if not isinstance(pbmc_atac2.X, np.ndarray):
            pbmc_atac2.X = pbmc_atac2.X.toarray()

        pbmc_atac2.X[pbmc_atac2.X>0.1]=1.0
        pbmc_atac2.X[pbmc_atac2.X<0.1]=0.0

        pbmc_atac2.X=scipy.sparse.csr_matrix(pbmc_atac2.X)
        #
        sc.pp.filter_genes(pbmc_atac2, min_cells=10)

        adata_utils.write_adata_as_10x_dir(pbmc_atac2, outdir='predict_filt_atac')






if __name__ == "__main__":
    main()