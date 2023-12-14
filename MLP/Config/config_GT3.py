class DefaultConfig(object):
    '''
    Configs for InferNet of GT3
    '''
    # tensorboard
    logs_str = r"GT3_InferNet"
    tag_str = r""

    ######*********CHECK CAREFULLY***********########
    seed = 4

    ARTIFICIAL = False      # If True, use the syns dataset; If False, use the real dataset.
    SET_VAL = False         # If False, 70% for training, 20% for testing; If True: the 20% is for vali, the 10% left is for testing
    DRAW_VAL = False        # Draw of not.

    USE_DA = False           # Use data augementation or not. This only work for REAL data

    noise_pct = 0.05        # noise percent in the synthetic data.

    ######*********CHECK CAREFULLY***********########

    train_pct = 0.7
    vali_pct = 0.2
    test_pct = 0.1

    # train and optim
    if ARTIFICIAL:
        # For artificial dataset
        batch_size = 56
        EPOCH_NUM= 30

        learning_rate = 5e-2
        lr_for_alpha = 5e-2

    else:
        # For real dataset
        batch_size = 56
        EPOCH_NUM = 30

        learning_rate = 5e-2
        lr_for_alpha = 5e-2

    lr_decay = 0.95      # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-2  # for optimizer (regularization)

    # For scheduler
    StepLR_step_size = 5
    StepLR_gamma = 0.9

    Exp_gamma = 0.98

    SAFETY = 1e-30

    SCALE = 1

    ################### DATA PATH ######################

    # Training data
    train_path = r"../data/train_8_all"

    if ARTIFICIAL:
        target_path_metric = "../data/artificial_targets_v2_" + "noise=" + str(noise_pct)
        target_path_loss = "../data/artificial_targets_v2_" + "noise=" + str(noise_pct) + "_ls_T"
    else:
        target_path_metric = "../data/targets_all"
        target_path_loss = "../data/targets_all_ls_T"

    # Target data for metric calculation
    if ARTIFICIAL:
        params_opitim_path = r"../data/auction_assign.csv"
    else:
        params_opitim_path = r"../data/SA_PT/params_opitim_delta_T.csv"

    # Target data for loss calculation
    TARGET = 1
    arr_flag = False        # whether drop uniform data

    # data keys
    data_key_path = "../data/target_datakey_all.csv"

    # NLL metric
    MIN_LOSS = 1e-30

    if ARTIFICIAL:
        NLL_metric_path = r"../data/GT_metric/NLL_metric_GT_Tgt=1_e30_artificial_v_2_noise="+str(noise_pct)+".csv"

    else:
        NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e30.csv"

    # Net path
    net_root_path = "net_saved/"