class DefaultConfig(object):
    '''
    Configs for using InferNet to generate predicted data.
    '''

    # tensorboard
    logs_str = r"GT_2_trial=1"
    tag_str = r""

    N_gaussians = 2

    GT_w_Params = 3  # GT2 or GT3

    seed = 3

    noise_pct = 0.05  # If True, use the syns dataset; If False, use the real dataset.

    ARTIFICIAL = False

    # Input NN path
    if ARTIFICIAL:
        model_params_MLP = r"../../MLP/net_saved/NN_params_infer_GT"+str(GT_w_Params)+"_artificial_v2_noise=" + str(noise_pct) + "_seed=" + str(seed) + ".pth"
    else:
        model_params_MLP = r"../../MLP/net_saved/NN_params_infer_GT"+str(GT_w_Params)+ "_seed=" + str(seed) + ".pth"

    # output path
    if ARTIFICIAL:
        params_gen_path = "../../data/SA_PT/params_artificial_GT"+str(GT_w_Params)+"_noise=" + str(noise_pct) + "_seed=" + str(seed) + ".csv"
    else:
        params_gen_path = "../../data/SA_PT/params_GT"+str(GT_w_Params)+"_seed=" + str(seed) + ".csv"


    ################### DATA PATH ######################

    # Training data
    train_path = r"../../data/train_8_all"

    #################################### The Below is not needed #####################


    batch_size = 40
    train_pct = 0.7
    vali_pct = 0.2
    test_pct = 0.1

    SAFETY = 1e-30

    SCALE = 1

    if ARTIFICIAL:
        target_path_metric = "../../data/artificial_targets_v2_" + "noise=" + str(noise_pct)
        target_path_loss = "../../data/artificial_targets_v2_" + "noise=" + str(noise_pct) + "_ls_T"

    else:
        target_path_metric = "../../data/artificial_targets_v2_" + "noise=" + str(noise_pct)
        target_path_loss = "../../data/artificial_targets_v2_" + "noise=" + str(noise_pct) + "_ls_T"


    # Target data for loss calculation
    TARGET = 1
    arr_flag = False        # whether drop uniform data

    # data keys
    data_key_path = "../../data/target_datakey_all.csv"

    # NLL metric
    MIN_LOSS = 1e-30
    NLL_metric_path = r"../../data/GT_metric/NLL_metric_GT_Tgt=1_e30_artificial_v_2_noise="+str(noise_pct)+".csv"
    params_opitim_path = r"../../data/auction_assign.csv"

