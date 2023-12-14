#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Description : Loss function. Metric calculation.

import torch


def cal_metric(Pi, Mu, Sigma, Duration, N_gaussians, vali_setting,MIN_LOSS,device):
    """
    Cal NLL metric of MDN
    Parameters
    ----------
    Pi
    Mu
    Sigma
    Duration
    N_gaussians
    vali_setting
    MIN_LOSS
    device

    Returns
    -------

    """
    NLL = torch.tensor(0.,device=device,requires_grad=True)
    for i in range(len(Pi)):
        target = Duration[i,:,0]
        pi = Pi[i,:]

        # Note: Mu == scale, Sigma = shape
        m = torch.distributions.Weibull(Mu[i,:], Sigma[i,:])

        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)
        target_nonzero = target[idx]
        target_nonzero_2 = target_nonzero.repeat(1,N_gaussians).to(device)

        loss_1 = (m.cdf(target_nonzero_2+0.5) - m.cdf(target_nonzero_2-0.5)).to(device=device)
        loss_2 = torch.sum(loss_1 * pi, dim=1)

        assert torch.all(loss_2)>=0,"in metric, loss_2<0"

        loss_3 = -torch.log(loss_2+MIN_LOSS)
        if torch.isnan(torch.sum(loss_3)):
            print("pi:",pi)
            print("Mu[i,:]:",Mu[i,:])
            print(" Sigma[i,:]:", Sigma[i,:])
            print("loss_1:",loss_1)
            print("loss_2:", loss_2)
            print("loss_3:",loss_3)

        NLL = torch.sum(loss_3)/len(idx) + NLL

    return NLL


def validate(mlp,data_loader,N_gaussians, MIN_LOSS, device, detail = False):
    total_metric = 0        # vali metric
    cnt = 0                 # vali set size
    odds_cnt_1 = 0          # times that NN behaves better than GT-1
    odds_cnt_2_common = 0          # times that NN behaves better than GT-2(common)
    odds_cnt_2_SA = 0          # times that NN behaves better than GT-2(SA)
    odds = [-1,-1,-1]            # winning rate of NN
    GT_metric = torch.tensor([0.,0.,0.,0.]).reshape(1,-1)

    for batch_id, data in enumerate(data_loader):
        input_data, target, _, setting , metric = data
        input_data = input_data.to(device)
        target = target.to(device)
        cnt = cnt + len(input_data)
        pi, mu, sigma = mlp(input_data)

        # Compute the error/ metric
        # Note: Mu == scale, Sigma = shape
        nll = cal_metric(pi, mu, sigma, target, N_gaussians, setting, MIN_LOSS, device)
        total_metric += nll.detach()

        # Sum up NLL of all vali data
        GT_metric += torch.sum(metric,dim=0)

        if detail:
            if nll.detach() < metric.detach()[0,0]:
                odds_cnt_1 += 1
            if nll.detach() < metric.detach()[0,1]:
                odds_cnt_2_common += 1
            if nll.detach() < metric.detach()[0,2]:
                odds_cnt_2_SA += 1

    # Get metric of GT model
    GT_metric = GT_metric/cnt

    total_metric = total_metric/cnt

    # Winning rate
    odds = [odds_cnt_1 / cnt, odds_cnt_2_common / cnt, odds_cnt_2_SA / cnt]
    return total_metric, GT_metric, odds


def loss_fn_wei(Pi, Mu, Sigma, Duration, N_gaussians, TARGET, eps, device):
    """
    Cal NLL loss when MDN uses Weibull distribution.
    Parameters
    ----------
    Pi
    Mu
    Sigma
    Duration
    N_gaussians
    eps: SAFETY

    Returns
    -------

    """
    loss_sum = torch.tensor(0.,device=device,requires_grad=True)

    for i in range(len(Pi)):

        target = Duration[i,:,0]
        pi = Pi[i,:]
        m = torch.distributions.Weibull(Mu[i,:],Sigma[i,:])

        # Drop padded data and Expanded to the same dim
        idx = torch.nonzero(target)
        target_nonzero = target[idx]
        target_nonzero_2 = target_nonzero.repeat(1,N_gaussians).to(device)

        loss_1 = (m.cdf(target_nonzero_2 + TARGET+0.5) - m.cdf(target_nonzero_2-0.5)).to(device=device)
        loss_2 = torch.sum(loss_1 * pi, dim=1)

        assert torch.all(loss_2) >= 0, "in loss, loss_2<0"

        loss_3 = torch.clamp(loss_2,min=eps)
        loss_4 = torch.log(loss_3)

        loss_sum = -torch.sum(loss_4) + loss_sum

    return (loss_sum)/len(Pi)



