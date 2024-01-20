

from loss_functions import *




def loss(preds, theta, truth, tb_step: int = None,
         scale_factor: float = 1.0,
         eps: float = 1e-10,
         mean: bool = True,
         debug: bool = False,
         tb: SummaryWriter = None,):
    """Calculates negative binomial loss as defined in the NB class in link above"""
    y_true = truth
    y_pred = preds * scale_factor

    if debug:  # Sanity check before loss calculation
        assert not torch.isnan(y_pred).any(), y_pred
        assert not torch.isinf(y_pred).any(), y_pred
        assert not (y_pred < 0).any()  # should be non-negative
        assert not (theta < 0).any()

    # Clip theta values
    theta = torch.clamp(theta, max=1e6)

    t1 = (
        torch.lgamma(theta + eps)
        + torch.lgamma(y_true + 1.0)
        - torch.lgamma(y_true + theta + eps)
    )
    t2 = (theta + y_true) * torch.log1p(y_pred / (theta + eps)) + (
        y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
    )
    if debug:  # Sanity check after calculating loss
        assert not torch.isnan(t1).any(), t1
        assert not torch.isinf(t1).any(), (t1, torch.sum(torch.isinf(t1)))
        assert not torch.isnan(t2).any(), t2
        assert not torch.isinf(t2).any(), t2

    retval = t1 + t2
    if debug:
        assert not torch.isnan(retval).any(), retval
        assert not torch.isinf(retval).any(), retval

    if tb is not None and tb_step is not None:
        tb.add_histogram("nb/t1", t1, global_step=tb_step)
        tb.add_histogram("nb/t2", t2, global_step=tb_step)

    return torch.mean(retval) if mean else retval

