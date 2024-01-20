"""
Custom activation functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Exp(nn.Module):
    """Applies torch.exp, clamped to improve stability during training"""

    def __init__(self, minimum=1e-5, maximum=1e6):
        """Values taken from DCA"""
        super(Exp, self).__init__()
        self.min_value = minimum
        self.max_value = maximum

    def forward(self, input):
        return torch.clamp(
            torch.exp(input),
            min=self.min_value,
            max=self.max_value,
        )


class ClippedSoftplus(nn.Module):
    def __init__(self, beta=1, threshold=20, minimum=1e-4, maximum=1e3):
        super(ClippedSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.min_value = minimum
        self.max_value = maximum

    def forward(self, input):
        return torch.clamp(
            F.softplus(input, self.beta, self.threshold),
            min=self.min_value,
            max=self.max_value,
        )

    def extra_repr(self):
        return "beta={}, threshold={}, min={}, max={}".format(
            self.beta,
            self.threshold,
            self.min_value,
            self.max_value,
        )
