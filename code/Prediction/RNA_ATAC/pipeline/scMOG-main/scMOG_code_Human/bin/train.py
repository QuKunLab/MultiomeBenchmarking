"""
Code to train the model
"""

import os
import sys
import logging
import argparse
import itertools
import numpy as np
import scipy.spatial
import collections
import matplotlib.pyplot as plt
import mpl_scatter_density

import torch
import torch.nn as nn
import torch.utils.data as Data
import sklearn.metrics as metrics
from astropy.visualization.mpl_normalize import ImageNormalize
import random
from astropy.visualization import LogStretch
from typing import *


# To ensure that the output is fixed when the same input
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


import both_GAN_1
import anndata as ad
import activations
import utils
import lossfunction
import losses
from pytorchtools import EarlyStopping


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
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
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

def plot_loss_history(history1,history2,history3,fname: str):
    """Constructing training loss curves"""
    fig, ax = plt.subplots(dpi=300)
    ax.plot(
        np.arange(len(history1)), history1, label="Train_G",
    )
    if len(history2):
        ax.plot(
        np.arange(len(history2)), history2, label="Train_D",)
    if len(history3):
        ax.plot(
        np.arange(len(history3)), history3, label="Test_G",)
    ax.legend()
    ax.set(
        xlabel="Epoch", ylabel="Loss",
    )
    #plt.show()
    fig.savefig(fname)
    return fig



def plot_auroc(
        truth,
        preds,
        title_prefix: str = "Receiver operating characteristic",
        fname: str = "",
):
    """
    Plot AUROC after flattening inputs
    """
    truth = truth.cpu().numpy().flatten()
    preds = preds.cpu().numpy().flatten()
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


def plot_prc(
        truth,
        preds,
        title_prefix: str = "Receiver operating characteristic",
        fname: str = "",
):
    """
    Plot PRC after flattening inputs
    """
    truth = truth.cpu().numpy().flatten()
    preds = preds.cpu().numpy().flatten()
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
    truth = truth.cpu().numpy().flatten()
    preds = preds.cpu().numpy().flatten()
    rmse=np.sqrt(metrics.mean_squared_error(truth, preds))
    logging.info(f"Found RMSE of {rmse:.4f}")

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
        x = np.log1p(x.cpu())
        y = np.log1p(y.cpu())

    # Ensure correct format
    x = x.cpu().numpy().flatten()
    y = y.cpu().numpy().flatten()
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
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")

    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    sc_rna_train_dataset=ad.read_h5ad('train_rna.h5ad')
    sc_atac_train_dataset=ad.read_h5ad('train_atac.h5ad')
    sc_rna_test_dataset=ad.read_h5ad('valid_rna.h5ad')
    sc_atac_test_dataset=ad.read_h5ad('valid_atac.h5ad')
    sc_rna_dataset=ad.read_h5ad('full_rna.h5ad')
    sc_atac_dataset=ad.read_h5ad('full_atac.h5ad')
    cuda = True if torch.cuda.is_available() else False
    device_ids = range(torch.cuda.device_count())

# Model
    param_combos = list(
        itertools.product(
            args.hidden,  args.lr,args.seed
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

        generatorATAC = both_GAN_1.GeneratorATAC(hidden_dim=h_dim,
                                           input_dim1=sc_rna_dataset.X.shape[1],
                                           input_dim2=sc_atac_dataset.X.shape[1],
                                           final_activations2=nn.Sigmoid(),
                                           flat_mode=True,
                                           seed=rand_seed,
                                           )
        generatorRNA = both_GAN_1.GeneratorRNA(hidden_dim=h_dim,
                                               input_dim1=sc_rna_dataset.X.shape[1],
                                               input_dim2=sc_atac_dataset.X.shape[1],
                                               final_activations1 = [activations.Exp(), activations.ClippedSoftplus()],
                                               flat_mode=True,
                                               seed=rand_seed,
                                               )



        RNAdiscriminator = both_GAN_1.Discriminator1(input_dim=sc_rna_train_dataset.X.shape[1],seed=rand_seed)
        ATACdiscriminator = both_GAN_1.Discriminator1(input_dim=sc_atac_train_dataset.X.shape[1],seed=rand_seed)

        # Loss function
        loss_bce = losses.FocalLoss_MultiLabel()
        loss_rna = lossfunction.loss

        def loss_D(fake,real,Discriminator):
            loss2_1 = -torch.mean(Discriminator(real))
            if isinstance(fake, tuple):
                loss2_2 = torch.mean(Discriminator(fake[0].detach()))
            else:
                loss2_2 = torch.mean(Discriminator(fake.detach()))
            loss2 = loss2_1 + loss2_2
            return loss2

        def loss_rna_G(fake,Discriminator):
            loss1 =-torch.mean(Discriminator(fake[0]))
            return loss1

        def loss_atac_G(fake,Discriminator):
            loss1 = -torch.mean(Discriminator(fake))
            return loss1


        if cuda:
           generatorRNA.cuda()
           generatorATAC.cuda()
           RNAdiscriminator.cuda()
           ATACdiscriminator.cuda()




        if len(device_ids) > 1:
            generatorRNA = torch.nn.DataParallel(generatorRNA)
            generatorATAC=torch.nn.DataParallel(generatorATAC)
            RNAdiscriminator = torch.nn.DataParallel(RNAdiscriminator)
            ATACdiscriminator = torch.nn.DataParallel(ATACdiscriminator)


        optimizer_rna_1 = torch.optim.Adam(generatorRNA.parameters(), lr=lr, betas=(args.b1, args.b2))
        optimizer_atac_1 = torch.optim.Adam(generatorATAC.parameters(), lr=lr, betas=(args.b1, args.b2))
        optimizer_rna = torch.optim.RMSprop(generatorRNA.parameters(), lr=lr)
        optimizer_atac = torch.optim.RMSprop(generatorATAC.parameters(), lr=lr)
        optimizer_D_rna = torch.optim.RMSprop(RNAdiscriminator.parameters(), lr=lr)
        optimizer_D_atac = torch.optim.RMSprop(ATACdiscriminator.parameters(), lr=lr)



        def pretrain_epoch(train_iter,generator,discriminator,updaterG,updaterD,lossG_history,lossD_history):
            generator.train()
            discriminator.train()
            train_losses=[]
            trainD_losses=[]
            for i, (x,y) in enumerate(train_iter):
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                updaterD.zero_grad()
                y_fake = generator(x)
                loss2=loss_D(y_fake,y,discriminator)
                loss2.backward()
                updaterD.step()
                trainD_losses.append(loss2.item())

                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

                if i % 5 == 0:
                    updaterG.zero_grad()
                    y_hat =generator(x)
                    if isinstance(y_hat, tuple):
                        loss1=loss_rna_G(y_hat,discriminator)
                    else:
                        loss1 = loss_atac_G(y_hat,discriminator)
                    loss1.backward()
                    updaterG.step()
                    train_losses.append(loss1.item())

            train_loss = np.average(train_losses[:-1])
            trainD_loss = np.average(trainD_losses[:-1])
            logging.info(f"lossG: {train_loss}")
            logging.info(f"lossD: {trainD_loss}")
            lossG_history.append(train_loss)
            lossD_history.append(trainD_loss)

            return lossG_history, lossD_history

        def training_epoch(train_iter, generator, updaterG,lossG_history):
            generator.train()
            train_losses = []
            for i, (x,y) in enumerate(train_iter):
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                updaterG.zero_grad()
                y_hat = generator(x)
                if isinstance(y_hat, tuple):
                    loss=loss_rna(preds=y_hat[0],theta=y_hat[1],truth=y)
                else:
                    loss = loss_bce(y_hat,y)
                loss.backward()
                updaterG.step()
                train_losses.append(loss.item())
            train_loss = np.average(train_losses[:-1])
            logging.info(f"AEloss: {train_loss}")
            lossG_history.append(train_loss)
            return lossG_history

        def test_epoch(generator,discriminator,test_iter,lossG_test_history):
            generator.eval()
            if discriminator:
                discriminator.eval()
            valid_losses = []
            with torch.no_grad():
                for (x,y)in test_iter:
                    if cuda:
                        x= x.cuda()
                        y= y.cuda()
                    y_hat =generator(x)
                    if discriminator:
                        if isinstance(y_hat, tuple):
                            loss=loss_rna_G(y_hat,discriminator)
                        else:
                            loss= loss_atac_G(y_hat,discriminator)
                    else:
                        if isinstance(y_hat, tuple):
                            loss = loss_rna(preds=y_hat[0], theta=y_hat[1], truth=y)
                        else:
                            loss = loss_bce(y_hat, y)
                    valid_losses.append(loss.item())

            valid_loss = np.average(valid_losses[:-1])
            logging.info(f"loss_test: {valid_loss}")
            lossG_test_history.append(valid_loss)
            return lossG_test_history,valid_loss

        def predict_atac(truth,generator,truth_iter):
            logging.info("....................................Evaluating ATAC ")
            def predict1(generator,truth_iter):
                generator.eval()
                first = 1
                for x in truth_iter:
                    if cuda:
                        x = x.cuda()
                    with torch.no_grad():
                        y_pred = generator(x)
                        if first == 1:
                            ret = y_pred
                            first = 0
                        else:
                            ret = torch.cat((ret, y_pred), 0)
                return ret

            sc_rna_atac_truth_preds = predict1(generator,truth_iter)
            fig = plot_auroc(
                truth,
                sc_rna_atac_truth_preds,
                title_prefix="RNA > ATAC",
                fname=os.path.join(outdir_name, f"rna_atac_auroc.{args.ext}"),
            )

            fig1 = plot_prc(
                truth,
                sc_rna_atac_truth_preds,
                title_prefix="RNA > ATAC",
                fname=os.path.join(outdir_name, f"rna_atac_auprc.{args.ext}"),
            )
            rmse_value(truth,sc_rna_atac_truth_preds)
            plt.close(fig)
            plt.close(fig1)

        def predict_rna(truth,generator,truth_iter):
            logging.info(".........................................Evaluating  RNA")

            def predict2(generator,truth_iter):
                generator.eval()
                first = 1
                for x in truth_iter:
                    if cuda:
                        x = x.cuda()
                    with torch.no_grad():
                        y_pred = generator(x)
                        if first == 1:
                            ret = y_pred[0]
                            first = 0
                        else:
                            ret = torch.cat((ret, y_pred[0]), 0)
                return ret

            sc_rna_truth_preds = predict2(generator,truth_iter)

            fig = plot_scatter_with_r(
                truth,
                sc_rna_truth_preds,
                one_to_one=True,
                logscale=True,
                density_heatmap=True,
                title="ATAC > RNA (test set)",
                fname=os.path.join(outdir_name, f"atac_rna_scatter_log.{args.ext}"),
            )
            rmse_value(truth,sc_rna_truth_preds)
            plt.close(fig)



        def train(generator,discriminator,num_epochs, train_iter,test_iter,truth_iter,truth,updaterG,updaterD,ISRNA):  # @save

            lossG_history = []
            lossD_history = []
            lossG_test_history=[]

            early_stopping = EarlyStopping(patience=7,verbose=True)
            for epoch in range(num_epochs):
                logging.info(f"....................................................this is epoch: {epoch}")
                if discriminator:
                    lossG_history,lossD_history=pretrain_epoch(train_iter,generator,discriminator,updaterG,updaterD,lossG_history,lossD_history)
                    if ((epoch + 1) %20== 0):
                        if ISRNA:
                            predict_rna(truth, generator, truth_iter)
                        else:
                            predict_atac(truth, generator, truth_iter)
                else:
                    lossG_history=training_epoch(train_iter,generator,updaterG,lossG_history)
                    if ((epoch + 1) % 5== 0):
                        if ISRNA:
                            predict_rna(truth, generator, truth_iter)
                        else:
                            predict_atac(truth, generator, truth_iter)

                if test_iter:
                    lossG_test_history,lossG_test=test_epoch(generator,discriminator,test_iter,lossG_test_history)

                early_stopping(lossG_test, generator)

                if early_stopping.early_stop:
                    logging.info("early stopping")
                    if ISRNA:
                        predict_rna(truth, generator, truth_iter)
                    else:
                        predict_atac(truth, generator, truth_iter)
                    break

            return lossG_history, lossD_history,lossG_test_history

#csr matrix to tensor
        Acoo = sc_rna_train_dataset.X.tocoo()
        sc_rna_train= scipy_sparse_mat_to_torch_sparse_tensor(Acoo)
        sc_rna_train=sc_rna_train.to_dense()

        Acoo1 = sc_atac_train_dataset.X.tocoo()
        sc_atac_train = scipy_sparse_mat_to_torch_sparse_tensor(Acoo1)
        sc_atac_train = sc_atac_train.to_dense()


        Acoo2 = sc_rna_test_dataset.X.tocoo()
        sc_rna_test = scipy_sparse_mat_to_torch_sparse_tensor(Acoo2)
        sc_rna_test = sc_rna_test.to_dense()

        Acoo3 = sc_atac_test_dataset.X.tocoo()
        sc_atac_test = scipy_sparse_mat_to_torch_sparse_tensor(Acoo3)
        sc_atac_test = sc_atac_test.to_dense()


        sc_atac_truth_dataset = ad.read_h5ad('truth_atac.h5ad')
        Acoo4 = sc_atac_truth_dataset.X.tocoo()
        sc_atac_truth = scipy_sparse_mat_to_torch_sparse_tensor(Acoo4)
        sc_atac_truth = sc_atac_truth.to_dense()

        sc_rna_truth_dataset = ad.read_h5ad('truth_rna.h5ad')
        Acoo3 = sc_rna_truth_dataset.X.tocoo()
        sc_rna_truth = scipy_sparse_mat_to_torch_sparse_tensor(Acoo3)
        sc_rna_truth = sc_rna_truth.to_dense()


        #Dataset Summary
        train_dataset1= Data.TensorDataset(sc_rna_train, sc_atac_train)
        train_iter1=torch.utils.data.DataLoader(dataset=train_dataset1,batch_size=256,shuffle=True)

        train_dataset2 = Data.TensorDataset( sc_atac_train,sc_rna_train)
        train_iter2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=64,shuffle=True)

        test_dataset1 = Data.TensorDataset(sc_rna_test, sc_atac_test)
        test_iter1 = torch.utils.data.DataLoader(dataset=test_dataset1, batch_size=64,)

        test_dataset2 = Data.TensorDataset(sc_atac_test, sc_rna_test)
        test_iter2 = torch.utils.data.DataLoader(dataset=test_dataset2, batch_size=64, )

        truth_iter_rna = torch.utils.data.DataLoader(dataset=sc_rna_truth, batch_size=64)
        truth_iter_atac = torch.utils.data.DataLoader(dataset=sc_atac_truth, batch_size=64)


        logging.info("...............................pretraining RNA -> ATAC")
        loss1_history, loss2_history, loss1_test_history = train(generator=generatorATAC,
                                                                 discriminator=ATACdiscriminator, num_epochs=200,
                                                                 train_iter=train_iter1,
                                                                 test_iter=test_iter1, truth_iter=truth_iter_rna,
                                                                 truth=sc_atac_truth,
                                                                 updaterG=optimizer_atac,
                                                                 updaterD=optimizer_D_atac, ISRNA=False)
        # loss visualization
        fig = plot_loss_history(
            loss1_history, loss2_history, loss1_test_history, os.path.join(outdir_name, f"losspretrain-ATAC.{args.ext}")
        )
        plt.close(fig)

        logging.info(
            "........................................................................................................................................................")
        logging.info("..............................pretraining ATAC -> RNA")
        loss1_history, loss2_history, loss1_test_history = train(generator=generatorRNA,
                                                                 discriminator=RNAdiscriminator, num_epochs=200,
                                                                 train_iter=train_iter2,
                                                                 test_iter=test_iter2, truth_iter=truth_iter_atac,
                                                                 truth=sc_rna_truth,
                                                                 updaterG=optimizer_rna,
                                                                 updaterD=optimizer_D_rna, ISRNA=True)
        # loss visualization
        fig = plot_loss_history(
            loss1_history, loss2_history, loss1_test_history, os.path.join(outdir_name, f"losspretrain-RNA.{args.ext}")
        )
        plt.close(fig)
        logging.info(
            "........................................................................................................................................................")

        logging.info("training ATAC -> RNA")
        loss1_history,loss2_history,loss1_test_history=train(generator=generatorRNA,
                                                             discriminator=None,num_epochs=60,
                                                             train_iter=train_iter2,
                                                             test_iter=test_iter2,truth_iter=truth_iter_atac,
                                                             truth=sc_rna_truth,
                                                             updaterG=optimizer_rna_1,
                                                             updaterD=None,ISRNA=True)
        # loss visualization
        fig = plot_loss_history(
            loss1_history, loss2_history, loss1_test_history, os.path.join(outdir_name, f"lossATAC-RNA.{args.ext}")
        )
        plt.close(fig)

        torch.save(generatorRNA.state_dict(),os.path.join(outdir_name, f"RNAgenerator.pth"))
        torch.save(RNAdiscriminator.state_dict(), os.path.join(outdir_name, f"RNAdiscriminator.pth"))

        logging.info("........................................................................................................................................................")
        logging.info("training RNA -> ATAC")
        loss1_history, loss2_history, loss1_test_history = train(generator=generatorATAC,
                                                                 discriminator=None, num_epochs=500,
                                                                 train_iter=train_iter1,
                                                                 test_iter=test_iter1, truth_iter=truth_iter_rna,
                                                                 truth=sc_atac_truth,
                                                                 updaterG=optimizer_atac_1,
                                                                 updaterD=None, ISRNA=False)
        # loss visualization
        fig = plot_loss_history(
            loss1_history, loss2_history, loss1_test_history, os.path.join(outdir_name, f"lossRNA-ATAC.{args.ext}")
        )
        plt.close(fig)



        ## save model

        torch.save(generatorATAC.state_dict(),os.path.join(outdir_name, f"ATACgenerator.pth"))
        torch.save(ATACdiscriminator.state_dict(), os.path.join(outdir_name, f"ATACdiscriminator.pth"))





if __name__ == "__main__":
    main()