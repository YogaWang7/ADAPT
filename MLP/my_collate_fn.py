#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : All functions of data collating are here.

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from MLP.Config.config import DefaultConfig
opt = DefaultConfig()

def my_collate_fn_3(data):

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, 
    target_loss_list = []       # target data for computing loss of NN, 
    setting_list = []
    metric_list = []

    data_len = len(data)        

    batch = 0

    while batch < data_len:
        # print("shape:",data[batch][0].shape) #shape: (3, 300)

        # Padding with zero
        data[batch][0][0,np.where(data[batch][0][0]==0)] = opt.SAFETY
        data[batch][0][1,np.where(data[batch][0][1]==0)] = opt.SAFETY

        # 归一化
        data[batch][0][0] = data[batch][0][0]/sum(data[batch][0][0])
        data[batch][0][1] = data[batch][0][1]/sum(data[batch][0][1])


        data_list.append(torch.tensor(data[batch][0]))
        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    #index = data[:][5]
    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor #, index


def my_collate_fn_4(data):

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, 
    target_loss_list = []       # target data for computing loss of NN, 
    setting_list = []
    metric_list = []

    data_len = len(data)        

    batch = 0

    while batch < data_len:

        data_list.append(torch.tensor(data[batch][0]))
        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor

def my_collate_fn_1GT(data, GT_CHOSEN):
    '''
    Only one GT model (GT_CHOSEN) is used to generate the input as MDN.
    :param data:
    :param GT_CHOSEN: GT_CHOSEN = 1 means GT1 is chosen.
    :return: GT1+EMD or GT2+EMD
    '''

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN
    target_loss_list = []       # target data for computing loss of NN
    setting_list = []
    metric_list = []

    data_len = len(data)

    batch = 0

    while batch < data_len:

        # Padding with zero
        data[batch][0][GT_CHOSEN,np.where(data[batch][0][GT_CHOSEN]==0)] = opt.SAFETY

        # Normalization
        data[batch][0][GT_CHOSEN] = data[batch][0][GT_CHOSEN]/sum(data[batch][0][GT_CHOSEN])

        # Concat with embedding
        data_tmp = np.array([data[batch][0][GT_CHOSEN,:],data[batch][0][2,:]])
        data_list.append(torch.tensor(data_tmp))

        # data_list.append(torch.tensor(data[batch][0]))
        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor

def my_collate_fn_woemd(data):
    '''
    No embedding is used.
    :param data:
    :return: GT1+GT2
    '''

    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, 
    target_loss_list = []       # target data for computing loss of NN,
    setting_list = []
    metric_list = []

    data_len = len(data)        

    batch = 0

    while batch < data_len:

        # Padding with zero
        data[batch][0][0,np.where(data[batch][0][0]==0)] = opt.SAFETY
        data[batch][0][1,np.where(data[batch][0][1]==0)] = opt.SAFETY

        # Normalization
        data[batch][0][0] = data[batch][0][0]/sum(data[batch][0][0])
        data[batch][0][1] = data[batch][0][1]/sum(data[batch][0][1])

        data_tmp = np.array([data[batch][0][0,:],data[batch][0][1,:]])
        data_list.append(torch.tensor(data_tmp))

        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor


def my_collate_fn_InferNet(data):
    '''
    InferNet
    '''
    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN
    target_loss_list = []       # target data for computing loss of NN

    target_params_list = []       # target data for computing loss of NN
    setting_list = []
    metric_list = []

    data_len = len(data)        

    batch = 0

    while batch < data_len:
        data_list.append(torch.tensor(data[batch][0]))

        target_metric_list.append(torch.tensor(data[batch][1][:,0], dtype=torch.int64))
        target_loss_list.append(torch.tensor(data[batch][2][:,0], dtype=torch.int64))
        target_params_list.append(torch.tensor(data[batch][3]))
        setting_list.append(torch.tensor(data[batch][4]))
        metric_list.append(torch.tensor(data[batch][5]))
        batch += 1


    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_tensor = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    target_params_tensor = torch.stack(target_params_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    ## Save some poor space
    del target_metric_padded,target_loss_padded, data_list, target_metric_list,target_loss_list,setting_list,metric_list
    return data_tensor, target_metric_tensor, target_loss_tensor, target_params_tensor, setting_tensor, metric_tensor


def my_collate_fn_0GT(data, which_dim):
    '''
    Only use data from GT1 or GT2 or EMB as input
    :param data:
    :param which_dim:
    :return: GT1 or GT2 or EMB
    '''
    data_list = []              # training data
    target_metric_list = []     # target data for computing metric of NN, 
    target_loss_list = []       # target data for computing loss of NN, 
    setting_list = []
    metric_list = []

    data_len = len(data)        

    batch = 0

    while batch < data_len:
        # GT1 or GT2
        if which_dim == 1 or which_dim == 0:
            # Padding with zero
            data[batch][0][which_dim,np.where(data[batch][0][which_dim]==0)] = opt.SAFETY
            data[batch][0][which_dim] = data[batch][0][which_dim]/sum(data[batch][0][which_dim])


        data_list.append(torch.tensor(data[batch][0][which_dim]).unsqueeze(0))
        target_metric_list.append(torch.tensor(data[batch][1]))
        target_loss_list.append(torch.tensor(data[batch][2]))
        setting_list.append(torch.tensor(data[batch][3]))
        metric_list.append(torch.tensor(data[batch][4]))
        batch += 1

    # Pad target data with zeros
    target_metric_padded = pad_sequence(target_metric_list,batch_first=True)
    target_loss_padded = pad_sequence(target_loss_list,batch_first=True)
    target_metric_tensor = target_metric_padded.float()
    target_loss_padded = target_loss_padded.float()

    data_tensor = torch.stack(data_list).float()
    setting_tensor = torch.stack(setting_list).float()
    metric_tensor = torch.stack(metric_list).float()

    return data_tensor, target_metric_tensor, target_loss_padded, setting_tensor, metric_tensor


