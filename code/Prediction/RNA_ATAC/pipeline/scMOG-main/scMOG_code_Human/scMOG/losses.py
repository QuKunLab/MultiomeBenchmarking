
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



# 针对 Multi-Label 任务的 Focal Loss
class FocalLoss_MultiLabel(nn.Module):
    def __init__(self, alpha=0.93, gamma=2,eps=1e-8,  size_average=True):
        super(FocalLoss_MultiLabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = eps

    def forward(self, pt, y):

        batch_loss = -((1 - self.alpha) * y*torch.pow(1-pt+ 1e-5, self.gamma)* torch.log(pt + 1e-5) + self.alpha * (1 - y)*torch.pow(pt+ 1e-5, self.gamma) * torch.log(1 - pt + 1e-5))
        loss=batch_loss.sum(axis=1).mean()
        # x_sigmoid = x
        # xs_pos = x_sigmoid
        # xs_neg = 1 - x_sigmoid
        # los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        # los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        # log_p = los_pos + los_neg
        #
        # alpha1=self.alpha * y
        # alpha0=(1-self.alpha)*(1-y)
        # alpha=alpha0+alpha1
        #
        # pt1 = xs_neg * y
        # pt0 = xs_pos * (1 - y)
        # pt= pt0 + pt1
        #batch_loss = -alpha * (torch.pow(pt, self.gamma)) * log_p

        # loss_pos=-(1 - self.alpha) * y * torch.pow(1 - pt, self.gamma) * torch.log(pt + 1e-5)
        # loss_neg= -self.alpha * (1 - y)*torch.pow(pt, self.gamma) * torch.log(1 - pt + 1e-5)

        # if self.size_average:
        #     loss = batch_loss.mean()
        # else:

        # loss = batch_loss.sum()
        # loss_pos=-alpha1 * (torch.pow((pt1), self.gamma)) * los_pos
        # loss_neg = -alpha0 * (torch.pow((pt0), self.gamma)) * los_neg
        # print(np.nonzero(alpha0).shape[0])
        # print(np.nonzero(alpha1).shape[0])
        # print(loss_pos.sum())
        # print(loss_neg.sum())
        # print(loss)
        return loss



class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        #x_sigmoid = torch.sigmoid(x)
        x_sigmoid =x
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

            loss = -loss.sum(axis=1).mean()
            #
            # loss_pos = -torch.pow(1 - pt0, one_sided_gamma) * los_pos
            # loss_neg = -torch.pow(1 - pt1, one_sided_gamma)* los_neg
            # print(np.nonzero(pt0).shape[0])
            # print(np.nonzero(pt1).shape[0])
            # print((loss_pos / np.nonzero(pt0).shape[0]).sum())
            # print((loss_neg / np.nonzero(pt1).shape[0]).sum())

        return loss


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


# class BinaryDSCLoss(torch.nn.Module):
#     r"""
#     Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
#     ("Dice Loss for Data-imbalanced NLP Tasks" paper)
#
#     Args:
#         alpha (float): a factor to push down the weight of easy examples
#         gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
#         reduction (string): Specifies the reduction to apply to the output:
#             ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
#             ``'mean'``: the sum of the output will be divided by the number of
#             elements in the output, ``'sum'``: the output will be summed.
#
#     Shape:
#         - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
#         - targets: `(N,c)`
#     """
#
#     def __init__(self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = "mean") -> None:
#         super().__init__()
#         self.alpha = alpha
#         self.smooth = smooth
#         self.reduction = reduction
#
#     def forward(self, logits, targets):
#         probs = logits
#         pos_mask = (targets == 1).float()
#         neg_mask = (targets == 0).float()
#         pos_weight = pos_mask * ((1 - probs) ** self.alpha) * probs
#         pos_loss = 1 - (2 * pos_weight + self.smooth) / (pos_weight + 1 + self.smooth)
#         #print(pos_loss)
#         neg_weight = neg_mask * ((1 - probs) ** self.alpha) * probs
#         neg_loss = 1 - (2 * neg_weight + self.smooth) / (neg_weight + self.smooth)
#         #print(neg_loss)
#         loss = pos_loss + neg_loss
#         #loss = loss.mean()
#         loss = loss.sum(axis=1).mean()
#         return loss

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        # assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        # predict = predict.contiguous().view(predict.shape[0], -1)
        # target = target.contiguous().view(target.shape[0], -1)
        #print(torch.sum(torch.mul(predict, target), dim=1) )
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class BCE_WITH_WEIGHT(torch.nn.Module):
    def __init__(self, alpha=0.05, reduction='mean'):
        super(BCE_WITH_WEIGHT, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = predict
        loss = -((1 - self.alpha) * target * torch.log(pt + 1e-5) + self.alpha * (1 - target) * torch.log(1 - pt + 1e-5))

        # print(np.nonzero(target).shape[0])
        # print(np.nonzero(1 - target).shape[0])

        loss = loss.sum(axis=1).mean()
        # loss_pos=-(1 - self.alpha) * target * torch.log(pt + 1e-5)
        # loss_neg= -self.alpha * (1 - target) * torch.log(1 - pt + 1e-5)
        # print(loss_pos.sum())
        # print(loss_neg.sum())
        # if self.reduction == 'mean':
        #     loss = torch.mean(loss)
        # elif self.reduction == 'sum':
        #     loss = torch.sum(loss)
        #loss = loss.mean()

        return loss




class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.5):
        '''
        bins: split to n bins
        alpha: hyper-parameter
        '''
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()
        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()
        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd
        #print(beta)


        return self._custom_loss(x, target, beta[bin_idx])

class GHMC_Loss(GHM_Loss):
    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return x.detach() - target


if __name__ == '__main__':
    no_of_classes = 2
    logits = torch.rand(4,5).float()
    print(logits)
    labels = torch.randint(0,no_of_classes, size = (4,5)).float()
    print(labels)
    dice_loss=GHMC_Loss(bins=10, alpha=0.5)
    loss=dice_loss(logits,labels)
    print(loss)


