"""
Model architecture
"""

import os
import sys

from typing import List, Tuple, Union, Callable


import torch
import torch.nn as nn


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import activations



torch.backends.cudnn.deterministic = True  # For reproducibility
torch.backends.cudnn.benchmark = False


cuda = True if torch.cuda.is_available() else False

class ATACEncoder(nn.Module):
    def __init__(self, num_inputs: int, num_units=16,seed: int = 182822,):
        super().__init__()
        torch.manual_seed(seed)  ##Seed the CPU to generate random numbers so that the results are deterministic

        self.num_inputs = num_inputs
        self.num_units = num_units

        self.encode0 = nn.Linear(self.num_inputs, 512)
        nn.init.xavier_uniform_(self.encode0.weight)
        self.bn0 = nn.BatchNorm1d(512)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

        self.encode1 = nn.Linear(512, 64)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.encode2 = nn.Linear(64, self.num_units)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.bn2 = nn.BatchNorm1d(num_units)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act0(self.bn0(self.encode0(x)))
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.bn2(self.encode2(x)))
        return x

class RNAEncoder(nn.Module):
    def __init__(self, num_inputs: int, num_units=16,seed: int = 182822,):
        super().__init__()
        torch.manual_seed(seed)  ##Seed the CPU to generate random numbers so that the results are deterministic
        self.num_inputs = num_inputs
        self.num_units = num_units

        self.encode0 = nn.Linear(self.num_inputs, 256)
        nn.init.xavier_uniform_(self.encode0.weight)
        self.bn0 = nn.BatchNorm1d(256)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

        self.encode1 = nn.Linear(256, 64)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.encode2 = nn.Linear(64, self.num_units)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.bn2 = nn.BatchNorm1d(num_units)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act0(self.bn0(self.encode0(x)))
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.bn2(self.encode2(x)))
        return x



class ATACDecoder(nn.Module):
    def __init__(
        self,
            num_outputs: int,
            num_units=16,
            # activation=nn.PReLU,
            final_activation=nn.Sigmoid,
            seed: int = 182822,
    ):
        super(ATACDecoder, self).__init__()
        torch.manual_seed(seed)  ##Seed the CPU to generate random numbers so that the results are deterministic
        self.num_outputs = num_outputs
        self.latent_dim = num_units

        self.decode1 = nn.Linear(self.latent_dim, 64)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.decode2 = nn.Linear(64, 512)
        nn.init.xavier_uniform_(self.decode2.weight)
        self.bn2 = nn.BatchNorm1d(512)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)


        self.decode3 = nn.Linear(512, self.num_outputs)
        nn.init.xavier_uniform_(self.decode3.weight)
        nn.init.constant_(self.decode3.bias,-2.0)
        self.final_activations = final_activation

    def forward(self, x):
        x = self.act1(self.bn1(self.decode1(x)))
        x = self.act2(self.bn2(self.decode2(x)))
        x = self.final_activations(self.decode3(x))

        return x

class ProteinDecoder(nn.Module):
    def __init__(
        self,
            num_outputs: int,
            num_units=16,
            # activation=nn.PReLU,
            final_activation=nn.Identity(),
            seed: int = 182822,
    ):
        super(ProteinDecoder, self).__init__()
        torch.manual_seed(seed)
        self.num_outputs = num_outputs
        self.latent_dim = num_units

        self.decode1 = nn.Linear(self.latent_dim, 64)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.decode2 = nn.Linear(64,self.num_outputs)
        nn.init.xavier_uniform_(self.decode2.weight)
        self.final_activations = final_activation

    def forward(self, x):
        x = self.act1(self.bn1(self.decode1(x)))
        x = self.final_activations(self.decode2(x))
        return x


class RNADecoder(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        num_units: int = 16,
        intermediate_dim: int = 64,
        activation=nn.LeakyReLU,
        final_activation=None,
        seed: int = 182822,

    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_outputs = num_outputs
        self.num_units = num_units

        self.decode1 = nn.Linear(self.num_units, 64)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.decode2 = nn.Linear(64, 256)
        nn.init.xavier_uniform_(self.decode2.weight)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.decode21 = nn.Linear(256, self.num_outputs)
        nn.init.xavier_uniform_(self.decode21.weight)
        self.decode22 = nn.Linear(256, self.num_outputs)
        nn.init.xavier_uniform_(self.decode22.weight)
        self.decode23 = nn.Linear(256, self.num_outputs)
        nn.init.xavier_uniform_(self.decode23.weight)

        self.final_activations = nn.ModuleDict()
        if final_activation is not None:
            if isinstance(final_activation, list) or isinstance(
                final_activation, tuple
            ):
                assert len(final_activation) <= 3
                for i, act in enumerate(final_activation):
                    if act is None:
                        continue
                    self.final_activations[f"act{i+1}"] = act
            elif isinstance(final_activation, nn.Module):
                self.final_activations["act1"] = final_activation
            else:
                raise ValueError(
                    f"Unrecognized type for final_activation: {type(final_activation)}"
                )

    def forward(self, x,size_factors=None):
        """include size factor here because we may want to scale the output by that"""
        x = self.act1(self.bn1(self.decode1(x)))
        x = self.act2(self.bn2(self.decode2(x)))
        retval1 = self.decode21(x)  # This is invariably the counts
        #retval1 = self.final_activations(retval1)
        if "act1" in self.final_activations.keys():
            retval1 = self.final_activations["act1"](retval1)
        if size_factors is not None:
            sf_scaled = size_factors.view(-1, 1).repeat(1, retval1.shape[1])
            retval1 = retval1 * sf_scaled  # Elementwise multiplication

        retval2 = self.decode22(x)
        if "act2" in self.final_activations.keys():
            retval2 = self.final_activations["act2"](retval2)

        retval3 = self.decode23(x)
        if "act3" in self.final_activations.keys():
            retval3 = self.final_activations["act3"](retval3)

        return retval1,retval2,retval3



class Inference(nn.Module):
    def __init__(self, num_inputs: int,final_activation=nn.Sigmoid,):
        super().__init__()
        self.num_inputs = num_inputs
        #self.final_activation=final_activation

        self.encode0 = nn.Linear(self.num_inputs, 128)
        nn.init.xavier_uniform_(self.encode0.weight)
        self.bn0 = nn.BatchNorm1d(128)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

        self.encode1 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.encode2 = nn.Linear(64,1)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.act2 = final_activation

    def forward(self, x):
        x = self.act0(self.bn0(self.encode0(x)))
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.encode2(x))
        return x

class Generator(nn.Module):
    def __init__(
            self,
            input_dim1: int,
            input_dim2: int,
            #out_dim: List[int],
            out_dim1: int,
            out_dim2:int,
            hidden_dim: int = 16,
            final_activations2=nn.Sigmoid(),
            final_activations1: list = [activations.Exp(), activations.ClippedSoftplus()],

            flat_mode: bool = True,  # Controls if we have to re-split inputs
            seed: int = 182822,
    ):
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        nn.Module.__init__(self)
        torch.manual_seed(seed)  ##为CPU设置种子用于生成随机数，以使得结果是确定的

        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1,
        self.input_dim2 = input_dim2,
        self.out_dim1 = out_dim1
        self.out_dim2 = out_dim2
        self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )
        self.RNAdecoder = RNADecoder(num_outputs=out_dim1, num_units=hidden_dim,final_activation=final_activations1)
        self.ATACencoder = ATACEncoder(num_inputs=input_dim2, num_units=hidden_dim)
        self.ATACdecoder = ATACDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)


    def forward(self, x):
        encoded1 = self.RNAencoder(x[0])
        encoded2 = self.ATACencoder(x[1])
        decoded11=self.RNAdecoder(encoded1)
        decoded12=self.ATACdecoder(encoded1)
        decoded21=self.RNAdecoder(encoded2)
        decoded22=self.ATACdecoder(encoded2)
        return decoded11,decoded12,decoded21,decoded22

class GeneratorATAC(nn.Module):
    def __init__(
            self,
            input_dim1: int,
            input_dim2: int,
            #out_dim: List[int],
            hidden_dim: int = 16,
            final_activations2=nn.Sigmoid(),

            flat_mode: bool = True,  # Controls if we have to re-split inputs
            seed: int = 182822,
    ):
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1,
        self.input_dim2 = input_dim2,
        self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )
        #self.RNAdecoder = RNADecoder(num_outputs=out_dim1, num_units=hidden_dim,final_activation=final_activations1)
        #self.ATACencoder = ATACEncoder(num_inputs=input_dim2, num_units=hidden_dim)
        self.ATACdecoder = ATACDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)


    def forward(self, x):
        encoded = self.RNAencoder(x)
        decoded=self.ATACdecoder(encoded)
        return decoded

class GeneratorProtein(nn.Module):
    def __init__(
            self,
            input_dim1: int,
            input_dim2: int,
            #out_dim: List[int],
            hidden_dim: int = 16,
            final_activations2=nn.Identity(),

            flat_mode: bool = True,  # Controls if we have to re-split inputs
            seed: int = 182822,
    ):
        nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1,
        self.input_dim2 = input_dim2,
        self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )
        self.ProteinDecoder = ProteinDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)


    def forward(self, x):
        encoded = self.RNAencoder(x)
        decoded=self.ProteinDecoder(encoded)
        return decoded


class GeneratorRNA(nn.Module):
    def __init__(
            self,
            input_dim1: int,
            input_dim2: int,
            #out_dim: List[int],
            hidden_dim: int = 16,
            final_activations1: list = [activations.Exp(), activations.ClippedSoftplus()],

            flat_mode: bool = True,  # Controls if we have to re-split inputs
            seed: int = 182822,
    ):
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1,
        self.input_dim2 = input_dim2,

        #self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )
        self.RNAdecoder = RNADecoder(num_outputs=input_dim1, num_units=hidden_dim,final_activation=final_activations1)
        self.ATACencoder = ATACEncoder(num_inputs=input_dim2, num_units=hidden_dim)
        #self.ATACdecoder = ATACDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)


    def forward(self, x):
        encoded = self.ATACencoder(x)
        decoded=self.RNAdecoder(encoded)
        return decoded



class Discriminator(nn.Module):
    def __init__(self,input_dim: int,seed: int = 182822,):
        super(Discriminator, self).__init__()
        torch.manual_seed(seed)

        self.model = nn.Sequential(
            # nn.Linear(input_dim, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_dim, 256),
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




class Discriminator1(nn.Module):
    def __init__(self,input_dim: int,seed: int = 182822,):
        super(Discriminator1, self).__init__()
        torch.manual_seed(seed)

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
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


class DiscriminatorProtein(nn.Module):
    def __init__(self,input_dim: int,seed: int = 182822,):
        super(DiscriminatorProtein, self).__init__()
        torch.manual_seed(seed)

        self.model = nn.Sequential(
            # nn.Linear(input_dim, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64,1)
            #nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)

        return y

# class GeneratorATAC(nn.Module):
#     def __init__(
#             self,
#             input_dim1: int,
#             input_dim2: int,
#             #out_dim: List[int],
#             hidden_dim: int = 16,
#             final_activations2=nn.Sigmoid(),
#
#             flat_mode: bool = True,  # Controls if we have to re-split inputs
#             seed: int = 182822,
#     ):
#         nn.Module.__init__(self)
#         torch.manual_seed(seed)  ##为CPU设置种子用于生成随机数，以使得结果是确定的
#
#         self.flat_mode = flat_mode
#         self.input_dim1 = input_dim1,
#         self.input_dim2 = input_dim2,
#         self.final_activations = final_activations2
#         self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )
#         #self.RNAdecoder = RNADecoder(num_outputs=out_dim1, num_units=hidden_dim,final_activation=final_activations1)
#         #self.ATACencoder = ATACEncoder(num_inputs=input_dim2, num_units=hidden_dim)
#         self.ATACdecoder = ATACDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)
#         self.inference = Inference(num_inputs=input_dim1, final_activation=final_activations2)
#         self.region_factors = torch.nn.Parameter(torch.zeros(self.input_dim2))
#         nn.init.uniform_(self.region_factors)
#
#
#     def forward(self, x):
#         encoded = self.RNAencoder(x)
#         decoded=self.ATACdecoder(encoded)
#         final=decoded*self.inference(x)
#         final=final*self.final_activations(self.region_factors)
#         return final