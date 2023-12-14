#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.cuda.amp import GradScaler,autocast

def save_grad(name):
    def hook(grad):
        print("*******")
        print(f"name={name}, grad={grad}")
    return hook

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
    return np.clip(f_Equi(t, v, d, b, alpha),eps,1-eps)  # eps < u < 1-eps

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


def get_nll_loss(T_target, U, LEN, eps = 1e-10):
    """
    Cal NLL loss for InferNet of GT3.
    Args:
        T_target:
        U:
        LEN:
        eps: 1e-10

    Returns:

    """
    nll = 0.0

    # For each observed duration T_target[idx]
    for idx in range(0,len(T_target)):
        # Only take into consideration situations: Recorded in U
        if T_target[idx] <= LEN :
            nll += ( np.sum( np.log( U[1:(T_target[idx])+1] ) ) + np.log(1-U[T_target[idx]+1]))
        else:   # Exceed the upper bound.
            nll += T_target[idx] * np.log(eps)
    return float(-nll)

def get_nll_meric(T_target, U, LEN,TARGET = 1,eps = 1e-30,q=1):
    """
    Cal NLL metric for InferNet of GT2.

    Args:
        T_target:
        U:
        LEN:
        TARGET:
        eps:

    Returns:
    NLL metric: 正值,并且已经除以sample数量
    """

    nll = 0.
    # Solve for P with length of LEN
    P = np.array([0.0] * (LEN + 1))
    P[0] = 0.0
    tmp = np.array([0.0] * (LEN + 3))
    tmp[0] = 1.0
    # Note: P[i][t] = U[i][1]*U[i][2]*...*(1-U[i][t+1])
    for t in range(1, len(P)):
        tmp[t] = tmp[t - 1] * U[t]
        P[t] = (1 - U[t + 1]) * tmp[t]

    # According to the target data, compute the NLL value
    P_dict = {}
    # Sum prob in every interval of TARGET
    for i in range(1, LEN + 1, TARGET):
        j = min(LEN, i + TARGET)
        P_dict[i] = np.sum(P[i:j])

    # nll_i = sum(logP1+logP2+...)
    # Sum up all prob if GT gives one
    if q==1:
        for i in range(0, len(T_target)):
            if T_target[i] in P_dict:
                nll += -np.log(P_dict[T_target[i]] + eps)
            else:
                nll += -np.log(0. + eps)
        nll = nll / len(T_target)

    return nll

def get_P(U,LEN):
    """
    Args:
        U:
        LEN:
        v:
        d:
        b:
    """
    # Solve for P with length of LEN
    P = np.array([0.0]*(LEN+1))
    P[0] = 0.0
    tmp = np.array([0.0]*(LEN+3))
    tmp[0] = 1.0

    for t in range(1,len(P)):
        tmp[t] = tmp[t-1]*U[t]
        P[t] = (1-U[t+1])*tmp[t]

    return P[1:]


