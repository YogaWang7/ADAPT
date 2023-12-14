class DefaultConfig(object):
    '''
    Configs for MBMDN (ADAPT) training. Also can be used for other models with easy modification.
    '''

    # Input Choice
    GT_1 = 0
    GT_2 = 1
    EMD = 2

    N_gaussians = 2  # nums of mixture components

    train_pct = 0.7
    vali_pct = 0.2
    test_pct = 0.1

    ######*********CHECK CAREFULLY***********########
    seed = 3

    # Which Model to be Chosen
    VALI_DETAIL = False            # If true: output details of validation for plot.
    Conv_1_4 = True               # If True: MB-MDN； If False MLP-MDN

    ARTIFICIAL = False       # If True, use the syns dataset; If False, use the real dataset.
    REVENUE = False          # If True: compare the revenue.

    SET_VAL = False         # If False, 70% for training, 20% for testing; If True: the 20% is for vali, the 10% left is for testing
    DRAW_VAL = False        # Draw Vali results not.
    PRINT_VAL = True        # Print or Not the vali result

    USE_DA = False           # Use data augementation or not. This only work for REAL data
    noise_pct = 0.05        # noise percent in the synthetic data.

    q = 1
    ######*********CHECK CAREFULLY***********########

    # For synthetic data
    if ARTIFICIAL and Conv_1_4:
        EPOCH_NUM = 25
        batch_size = 48

        learning_rate = 5e-3
        lr_for_pi = 5e-3

        lr_for_mu = 1e-2
        lr_for_sigma = 1e-3

        lr_for_shape = 5e-3
        lr_for_scale = 5e-3

        # For scheduler
        StepLR_step_size = 5
        StepLR_gamma = 0.92

    # For real data
    if not ARTIFICIAL and Conv_1_4:

        EPOCH_NUM = 25

        batch_size = 40
        learning_rate = 1e-3

        lr_for_pi = 5e-3

        lr_for_mu = 1e-2
        lr_for_sigma = 5e-3

        # Weibull
        lr_for_shape = 5e-3
        lr_for_scale = 5e-3

        # For scheduler
        StepLR_step_size = 5
        StepLR_gamma = 0.92


    lr_decay = 0.95      # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-2  # for optimizer (regularization)

    SAFETY = 1e-30

    # For plotting
    SCALE = 1

    ################### DATA PATH ######################

    # Data path
    if ARTIFICIAL:
        train_path = "../data/artificial_train_v2_noise=" + str(noise_pct) + "_seed=" + str(seed)

        target_path_metric = "../data/artificial_targets_v2_" + "noise=" + str(noise_pct)
        target_path_loss = "../data/artificial_targets_v2_" + "noise=" + str(noise_pct) + "_ls_T"

        NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e30_all_" + "artificial_targets_v2_" + "noise=" + str(
        noise_pct) + "_seed=" + str(seed) + ".csv"
        # output: Net saving path
        net_root_path = "net_saved/MDN_artificial_seed=" + str(seed) + ".pth"

        # output: Details of validatiaon test
        vali_root = "../data/figs/vali_syn/vali_noise=" + str(noise_pct) + "_seed=" + str(seed)

        # Visdom env
        env_str = "synthetic_seed="+str(seed)

        # Tensorboard
        logs_str = "synthetic_seed=" + str(seed)
        vali_main_tag_str = "vali_metric/synthetic_seed=" + str(seed)

    else:
        train_path = "../data/train_300_uniq_all_seed=" + str(seed)
        target_path_metric = "../data/targets_all"

        if USE_DA:
            target_path_loss = "../data/targets_all_5_DA_P=0.5_N_c=2"
        else:
            target_path_loss = "../data/targets_all"

        NLL_metric_path = "../data/GT_metric/NLL_metric_GT_Tgt=1_e30_seed="+str(seed)+".csv"

        # output: Net saving path
        net_root_path = "net_saved/MDN_seed=" + str(seed) + ".pth"

        # output: Details of validatiaon test
        vali_root = "../data/figs/vali_real/vali_seed="+ str(seed)

        # Visdom env
        env_str = "real_seed="+str(seed)

        # Tensorboard
        logs_str = f"real_seed=" + str(seed)
        vali_main_tag_str = "vali_metric/real_seed=" + str(seed)

    # Target data for loss calculation
    TARGET = 1
    arr_flag = False        # whether drop uniform data
    arr_path = r"../data_handler/idx_GT2_better.pickle"

    # data keys
    data_key_path = "../data/target_datakey_all.csv"

    # Cluster assignment
    N_CLUSTER = 7
    CHOSEN_CLUSTER = 2
    cluster_assign_path = "../data/cluster_assignment_N="+str(N_CLUSTER) + "CHOSEN="+str(CHOSEN_CLUSTER)+".csv"

    MIN_LOSS = 1e-30