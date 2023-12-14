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


# prob. weighting func。根据Eq(5)
def OMEGA(p):
    '''
    prob. weighting func

    :param p:
    :return:
    '''
    return p


def C(t,b):
    '''
    Sunk cost function

    :param t:
    :param b:
    :return:
    '''
    return 0.2*t*b

def f(x, alpha):
    '''
    utility function = f = 1-f_2

    :param x:
    :param alpha:
    :return:
    '''
    return (1-np.exp(-alpha*x))

def f_2(x,alpha):
    '''
    utility function = f - 1-f_2

    :param x:
    :param alpha:
    :return:
    '''

    return np.exp(-alpha*x)
def f_Equi(t,v,d,b,alpha,labda):
    '''
        Returns: u_t (not log(u_t))

    :param t:
    :param v:
    :param d:
    :param b:
    :param alpha:
    :param labda:
    :return:
    '''

    tmp = v - d * (t - 1) - C(t - 2, b) - b
    if (tmp>0):

        root = (labda*f(C(t-2,b),alpha) + f(tmp,alpha)) / (labda*f(C(t-2,b)+b,alpha) + f(tmp,alpha))

    else:

        root = (1- f_2(-(v-(t-1)*d-b),alpha)) / (f_2(b,alpha) - f_2(-(v-(t-1)*d-b),alpha))


    assert root >= 0, f"Before clip, U[t] < 0 , tmp = {tmp}, U[t] = {root}, t,v,d,b,alpha,labda = {t,v,d,b,alpha,labda}"
    assert root <= 1, f"Before clip, U[t] > 1 , tmp = {tmp}, U[t] = {root}, t,v,d,b,alpha,labda = {t,v,d,b,alpha,labda}"

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


def U_GT2(t,v, d, b,alpha = -0.0135, labda = 3.3124,eps = 1e-10):
    return np.clip(f_Equi(t, v, d, b, alpha, labda),eps,1-eps)  # eps < U < 1-eps

def get_U_GT2(LEN,v, d, b, alpha = -0.0135, labda = 3.3124,eps = 1e-10):
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
    U_GT2_vec = np.vectorize(U_GT2)
    U[2:-1] = U_GT2_vec(t, v, d, b,alpha,labda,eps)

    # U[-1] exceeds the upper bound of T_i. Not caculable and set as zero.
    U[-1] = 0

    return U


def U_GT1(t, v, d, b,eps = 1e-10):
    return np.clip(1 - b / (v - (t - 1) * d), eps, 1 - eps)  # Make sure eps < U < 1-eps to prevent log(0)


def get_U_GT1(LEN,v, d, b,eps = 1e-10):
    U = [0] * (LEN + 2)  # U: the prob. that someone offers a bid in t_th round
    U[0], U[1] = 1., 1.  # Actually we do not need u[0]. U[1]=1 ensures that

    # Vectorize this calculation
    t = np.arange(2,LEN+1)
    U_GT1_vec = np.vectorize(U_GT1)
    U[2:-1] = U_GT1_vec(t, v, d, b,eps)

    # U[-1] exceeds the upper bound of T_i. Not caculable and set as zero.
    U[-1] = 0
    # U1[t] = 0
    # assert U == U1, "U != U1"
    return U
def get_nll_loss(T_target, U, LEN, eps = 1e-10):
    """
    Cal NLL loss for InferNet of GT2.
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
        # Note: Not all actual duration values can be generated out by GT models.
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
    NLL metric:
    """

    nll = 0.
    # Solve for P with length of LEN
    P = np.array([0.0] * (LEN + 1))
    P[0] = 0.0
    tmp = np.array([0.0] * (LEN + 3))
    tmp[0] = 1.0
    # Note：P[i][t] = U[i][1]*U[i][2]*...*(1-U[i][t+1])
    for t in range(1, len(P)):
        tmp[t] = tmp[t - 1] * U[t]  # tmp[t] is the continued product from U[1] to U[t].
        P[t] = (1 - U[t + 1]) * tmp[t]

    # According to the target data, compute the NLL
    P_dict = {}  # Dict is convenient for later calculation.

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
    :return: P
    """
    # Solve for P with length of LEN
    P = np.array([0.0]*(LEN+1))
    P[0] = 0.0
    tmp = np.array([0.0]*(LEN+3))
    tmp[0] = 1.0

    # Note: P[i][t] = U[i][1]*U[i][2]*...*(1-U[i][t+1])
    for t in range(1,len(P)):
        tmp[t] = tmp[t-1]*U[t]
        P[t] = (1-U[t+1])*tmp[t]

    return P[1:]
