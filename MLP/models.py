#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : All models are included here.


import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
bound_alpha = torch.tensor([-0.3,0.3],device=device)
bound_labda = torch.tensor([0.01,18],device=device)


class Conv_block_4(nn.Module):
    '''
    Block of MB-MDN model
    '''
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self, ch_out=1,kernel_size=9, stride=3, init_weight=True) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-kernel_size)/stride+1)

        self.ac_func = nn.Softplus()

        self.conv = nn.Conv2d(in_channels=1, out_channels=ch_out, kernel_size=self.kernel_size, stride=self.stride, padding=0, dilation=(1,1))

        self.BN_aff2 = nn.BatchNorm1d(num_features=self.ln_in,affine=True)      # works better

        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        # Conv=>BN=>AC
        x = self.conv(x)

        x = torch.flatten(x,start_dim=1)
        x = self.ac_func(self.BN_aff2(x))

        return x

class Conv_1_4(nn.Module):
    '''
    Basic MB-MDN model
    '''
    # code->generate->override methods
    def __init__(self, n_gaussians, ch_out=1, kernel_size=9, stride=3) -> None:
        super().__init__()

        self.kernel_size = (3,kernel_size)
        self.stride = (3,stride)
        self.ln_in = int((300-self.kernel_size[1])/self.stride[1]+1)

        self.BN1 = nn.BatchNorm1d(num_features=3,affine=True)

        self.layer_pi = Conv_block_4(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_scale = Conv_block_4(ch_out=1,kernel_size=kernel_size,stride=stride)
        self.layer_shape = Conv_block_4(ch_out=1,kernel_size=kernel_size,stride=stride)

        self.ac_func = nn.Softplus()

        self.z_pi = nn.Sequential(
            nn.Linear(self.ln_in, n_gaussians),
            nn.Softmax(dim=1)
        )
        self.z_scale = nn.Linear(self.ln_in, n_gaussians)
        self.z_shape = nn.Linear(self.ln_in, n_gaussians)

    def forward(self, x):

        x = self.BN1(x)
        x = torch.unsqueeze(x,dim=1)                     # torch.Size([B, 1, 3, 300])

        x_pi = self.layer_pi(x)
        x_scale = self.layer_scale(x)
        x_shape = self.layer_shape(x)

        pi = self.z_pi(x_pi)
        scale = torch.exp(self.z_scale(x_scale))
        scale = torch.clamp(scale,1e-4)
        shape = torch.exp(self.z_shape(x_shape))
        shape = torch.clamp(shape,1e-4)

        return pi,scale,shape

class MLP_1_1(nn.Module):
    '''
    2-layer InferNet for GT2.

    '''
    # code->generate->override methods
    def __init__(self) -> None:
        super().__init__()
        self.BN1 = nn.BatchNorm1d(num_features=8, affine=True)

        self.block_alpha = nn.Sequential(
            nn.Linear(8, 1),
            nn.Tanh()
        )
        self.block_labda = nn.Sequential(
            nn.Linear(8, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = torch.squeeze(x, dim=1)  # [B,C,N]->[B,N]，因为C=1
        x = self.BN1(x)

        alpha = self.block_alpha(x)
        labda = self.block_labda(x)

        # Clamp
        alpha = torch.clamp(alpha, min=bound_alpha[0], max=bound_alpha[1])
        labda = torch.clamp(labda, min=bound_labda[0], max=bound_labda[1])

        return alpha, labda

class MLP_2_1(nn.Module):
    '''
    3-layer InferNet for GT2.

    '''
    # code->generate->override methods
    def __init__(self) -> None:
        super().__init__()
        self.BN1 = nn.BatchNorm1d(num_features=8,affine=True)
        self.LN1 = nn.Linear(8,4)

        self.block_alpha = nn.Sequential(
            nn.Linear(4, 1),
            nn.Tanh()
        )
        self.block_labda = nn.Sequential(
            nn.Linear(4, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = torch.squeeze(x,dim=1)
        x = self.BN1(x)

        x = self.LN1(x)
        x = F.relu(x)

        alpha = self.block_alpha(x)
        labda = self.block_labda(x)

        # Clamp
        alpha = torch.clamp(alpha,min=bound_alpha[0],max=bound_alpha[1])
        labda = torch.clamp(labda,min=bound_labda[0],max=bound_labda[1])

        return alpha,labda

class MLP_GT3_1(nn.Module):
    '''
    2-layer InferNet for GT3.

    '''
    # code->generate->override methods
    def __init__(self) -> None:
        super().__init__()
        self.BN1 = nn.BatchNorm1d(num_features=8, affine=True)

        self.block_alpha = nn.Sequential(
            nn.Linear(8, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.BN1(x)

        alpha = self.block_alpha(x)

        # Clamp
        alpha = torch.clamp(alpha, min=bound_alpha[0], max=bound_alpha[1])

        return alpha

class MLP_GT3_2(nn.Module):
    '''
    3-layer InferNet for GT3.

    '''
    # code->generate->override methods
    def __init__(self) -> None:
        super().__init__()
        self.BN1 = nn.BatchNorm1d(num_features=8, affine=True)
        self.LN1 = nn.Linear(8, 4)

        self.block_alpha = nn.Sequential(
            nn.Linear(4, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.BN1(x)

        x = self.LN1(x)
        x = F.relu(x)

        alpha = self.block_alpha(x)

        # Clamp
        alpha = torch.clamp(alpha, min=bound_alpha[0], max=bound_alpha[1])

        return alpha

