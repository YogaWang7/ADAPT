#!/usr/bin/env python
# -*- coding: utf-8 -*-


from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np


class myDataset(Dataset):
    def __init__(self, train_path, target_path_metric, target_path_loss, key_path, metric_path):
        """
        target_path_metric
        target_path_loss
        """
        self.train_root_path = train_path
        self.target_metric_root_path = target_path_metric
        self.target_loss_root_path = target_path_loss

        self.train_all_path = os.listdir(train_path)
        self.target_all_path = os.listdir(target_path_metric)
        self.key_path = key_path
        self.metric_path = metric_path


    def __len__(self):
        """
        :return: the num of files in the dataset
        """
        return len(self.target_all_path)

    def __getitem__(self, index):
        """
        :param index:
        :return: train_data, target_data
        """
        train_path_i_path = os.path.join(self.train_root_path,self.train_all_path[index])
        target_loss_path_i_path = os.path.join(self.target_loss_root_path,self.target_all_path[index])
        target_metric_i_path = os.path.join(self.target_metric_root_path,self.target_all_path[index])

        train_df = pd.read_csv(train_path_i_path,encoding="utf-8")
        target_loss_df = pd.read_csv(target_loss_path_i_path,encoding="utf-8")
        target_metric_df = pd.read_csv(target_metric_i_path,encoding="utf-8")

        settings = pd.read_csv(self.key_path,encoding="utf-8")
        metric = pd.read_csv(self.metric_path,encoding="utf-8")

        settings_df = settings.iloc[index,1:]         # 'desc' is hard to handle bcs it's str.
        metric_df = metric.iloc[index,:]

        # Transform into numpy (not tensor!)
        train_data = np.array(train_df.values,dtype=float)
        target_loss_data = np.array(target_loss_df.values,dtype=float)
        target_metric_data = np.array(target_metric_df.values,dtype=float)

        settings_data = np.array(settings_df.values,dtype=float)
        metric_data = np.array(metric_df.values,dtype=float)


        return train_data, target_metric_data, target_loss_data, settings_data, metric_data #, index

