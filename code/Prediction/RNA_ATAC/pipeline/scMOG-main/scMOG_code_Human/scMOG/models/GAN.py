"""

"""

import os
import sys
import logging
from typing import List, Tuple, Union, Callable
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import skorch
import skorch.utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import activations
import model_utils


torch.backends.cudnn.deterministic = True  # For reproducibility
torch.backends.cudnn.benchmark = False


cuda = True if torch.cuda.is_available() else False


class Encoder(nn.Module):
    def __init__(self, num_inputs: int, num_units=16):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_units = num_units

        self.encode1 = nn.Linear(self.num_inputs, 64)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.encode2 = nn.Linear(64, self.num_units)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.bn2 = nn.BatchNorm1d(num_units)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.bn2(self.encode2(x)))
        return x

class ChromDecoder(nn.Module):
    """
     具有每染色体感知能力的网络，但不输出每染色体值，而是将它们连接成单个向量
    """

    def __init__(
        self,
        num_outputs: List[int],  # Per-chromosome list of output sizes
        latent_dim=16,
        #activation=nn.PReLU,
        final_activation=nn.Sigmoid,
    ):
        super(ChromDecoder, self).__init__()
        self.num_outputs = num_outputs
        self.latent_dim = latent_dim

        self.decode1 = nn.Linear(self.latent_dim, len(self.num_outputs) * 16)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(len(self.num_outputs) * 16)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.final_activations = final_activation


        self.final_decoders = nn.ModuleList()  # List[List[Module]]
        for n in self.num_outputs:
            layer0 = nn.Linear(16, 32)
            nn.init.xavier_uniform_(layer0.weight)
            bn0 = nn.BatchNorm1d(32)
            act0 = nn.LeakyReLU(0.2, inplace=True)
            layer1 = nn.Linear(32, n)
            nn.init.xavier_uniform_(layer1.weight)
            self.final_decoders.append(
                nn.ModuleList([layer0, bn0, act0, layer1])
            )

    def forward(self, x):
        x = self.act1(self.bn1(self.decode1(x)))
        # This is the reverse operation of cat
        x_chunked = torch.chunk(x, chunks=len(self.num_outputs), dim=1)

        first=1
        for chunk, processors in zip(x_chunked, self.final_decoders):
            # decode1, bn1, act1, *output_decoders = processors
            decode1, bn1, act1, decode2= processors
            chunk = act1(bn1(decode1(chunk)))
            temp= decode2(chunk)
            temp= self.final_activations(temp)
            if first==1:
                retval=temp
                first=0
            else:
                retval = torch.cat((retval,temp), dim=1)
        return retval

class Generator(nn.Module):
    def __init__(
            self,
            input_dim: int,
            out_dim: List[int],
            hidden_dim: int = 16,
            final_activations=nn.Sigmoid(),
            flat_mode: bool = True,  # Controls if we have to re-split inputs
            seed: int = 182822,
    ):
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        nn.Module.__init__(self)
        torch.manual_seed(seed)  ##为CPU设置种子用于生成随机数，以使得结果是确定的

        self.flat_mode = flat_mode
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.encoder = Encoder(num_inputs=input_dim, num_units=hidden_dim)

        self.decoder = ChromDecoder(
            num_outputs=out_dim,
            latent_dim=hidden_dim,
            final_activation=final_activations,
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded









class Discriminator(nn.Module):
    def __init__(self,input_dim: int,):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16,1)
            #nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)

        return y
