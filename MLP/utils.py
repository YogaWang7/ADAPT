# Commonly used funcs.
import pandas as pd
import torch
import numpy as np
import random

def setup_seed(seed):
    """
    Set seed
    Args:
        seed:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_data_idx(dataset,opt):
    """
    Args:
        dataset:
        opt:
    """
    shuffled_indices = []
    if not opt.arr_flag:
        DATA_len = dataset.__len__()
        shuffled_indices = np.random.permutation(DATA_len)

    if opt.arr_flag:
        shuffled_indices = np.load(opt.arr_path)
        np.random.shuffle(shuffled_indices)

    return shuffled_indices
def get_data_idx(shuffled_indices,opt):
    """
    To get data split idx according to shuffled 'shuffled_indices'
    Args:
        shuffled_indices:
        opt:

    Returns:
    """

    DATA_len = len(shuffled_indices)

    train_idx = shuffled_indices[:int(opt.train_pct * DATA_len)]
    if opt.SET_VAL:
        tmp = int((opt.train_pct + opt.vali_pct) * DATA_len)
        val_idx = shuffled_indices[int(opt.train_pct * DATA_len):tmp]
        test_idx = shuffled_indices[tmp:]
    else :
        # Exchange. 20% for testing
        tmp = int((opt.train_pct + opt.vali_pct) * DATA_len)
        test_idx = shuffled_indices[int(opt.train_pct * DATA_len):tmp]
        val_idx = shuffled_indices[tmp:]
    return train_idx,val_idx,test_idx


def get_data_idx_bidfee(shuffled_indices, opt, FEE=0.01):
    """
    To get data split idx according to shuffled 'shuffled_indices'
    Args:
        shuffled_indices:
        FEE: bid fee selected as non-training set
        opt:

    Returns:
    """
    data_key = pd.read_csv(opt.data_key_path)

    non_test_idx = [i for i in range(0,len(data_key)) if data_key.loc[i,'bidfee'] != FEE]

    test_idx = [i for i in range(len(data_key)) if i not in non_test_idx]
    print(f"testing set size = {len(test_idx)}")

    tmp = int((opt.vali_pct) * len(non_test_idx))
    val_idx = non_test_idx[-tmp:]
    train_idx = [i for i in non_test_idx if i not in val_idx]

    del data_key
    return train_idx,val_idx,test_idx

def get_data_idx_bidinc(shuffled_indices,opt,INC=0.01):
    """
    To get data split idx according to shuffled 'shuffled_indices'
    Args:
        shuffled_indices:
        FEE: bid fee selected as non-training set
        opt:

    Returns:
    """
    data_key = pd.read_csv(opt.data_key_path)

    non_test_idx = [i for i in range(0,len(data_key)) if data_key.loc[i,'bidincrement'] != INC]

    test_idx = [i for i in range(len(data_key)) if i not in non_test_idx]
    print(f"testing set size = {len(test_idx)}")

    # Vali set is 10%
    tmp = int((opt.vali_pct) * len(non_test_idx))
    val_idx = non_test_idx[-tmp:]
    train_idx = [i for i in non_test_idx if i not in val_idx]


    del data_key
    return train_idx,val_idx,test_idx







