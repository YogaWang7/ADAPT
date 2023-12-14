#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch

def f(x, alpha):
    """
    utility function = f = 1-f_2
    Args:
        x:
        alpha:

    Returns:

    """
    return (1-np.exp(-alpha*x))

def f_2(x,alpha):
    """
    utility function = f - 1-f_2
    Args:
        x:
        alpha:

    Returns:

    """
    # y = torch.min(-alpha*x,torch.tensor(85.))       # torch.exp(x), x<85
    # c = (-alpha*x).astype(np.float128)
    return np.exp(-alpha*x)

def f_Equi(t,v,d,b,alpha):
    '''

    Returns: u_t (not log(u_t))
    '''

    tmp = v - d * (t - 1) - b
    root = 1- (f_2(b,alpha)-1) / (f_2(tmp,alpha)-1)

    return root


def get_LEN_T(v,b,d,duration_max):
    """
    Cal LEN and T
    Args:
        v:
        b:
        d:
        T_i:

    Returns:
            LEN: U length aka. max duration GT should calculate
            T: duration limitation

    """

    T = 0
    LEN = 0

    if d == 0:      # fixed-price
        T = np.inf
        LEN = int(duration_max)
    else:           # asc-price
        T = np.floor((v - b) / d)
        LEN = int(T)

    return LEN, T

def U_GT3(t,v, d, b,alpha = -0.0135, eps = 1e-10):
    return np.clip(f_Equi(t, v, d, b, alpha),eps,1-eps)

def get_U_GT3(LEN,v, d, b, alpha = -0.0135,eps = 1e-10):
    """
    Cal U. Note: eps < U[t] < 1-eps, when 1 <= t <= LEN, and U[LEN+1] = 0
    Args:
        LEN:
        v:
        d:
        b:
        alpha:
        labda:
        eps = 1e-10. Set min(U)==eps to prevent log(0)

    Returns:
        U: prob vector and U[t] represents the prob that someone bids at period 't'

    """

    U = [0] * (LEN + 2)  # U: the prob. that someone offers a bid in t_th round
    U[0], U[1] = 1., 1.

    # Vectorize this calculation
    t = np.arange(2,LEN+1)
    U_GT3_vec = np.vectorize(U_GT3)
    U[2:-1] = U_GT3_vec(t, v, d, b,alpha,eps)

    # U[-1] exceeds the upper bound of T_i. Not caculable and set as zero.
    U[-1] = 0

    return U


