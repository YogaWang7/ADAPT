
- This is for _Predicting Real-World Penny Auction Durations by Integrating Game Theory and Machine Learning_
-----

# data
1. `target_datakey_all`: all (raw) auction configurations are here.
2. `targets_all`: all target data
3. `train_300_uniq_all_seed=3`: example of input of MB-MDN when seed = 3. When seed changes, the input data will actually change accordingly due to GT2.

# GT_Models
## GT1
1. In Stage II of ADAPT, the output (e.g., prediction) of GT1 is regarded as input of MB-MDN in Stage III.
2. `GT_1_gen` is to generate prediction of GT1.

## GT2
1. In Stage II of ADAPT, the output (e.g., prediction) of GT2 is regarded as input of MB-MDN in Stage III.
2. `SA_for_GT2_funcs.py`: infer parameter values of GT2 via Simulated Annealing.
3. `GT_2_get_params`: generate parameter values of GT2 via InferNet.
4. `GT_2_gen_uniq_params`: generate prediction of GT2 with parameter values generated by InferNet.

## GT3
1. GT3 is the game theory model we use to compare with other methods. The output (e.g., prediction) of GT3 is NOT regarded as input of MB-MDN in Stage III.
2. `GT_3_get_params`: generate parameter values of GT3 via InferNet.

# MLP
1. `auction_features_encoding`: encoding description.
2. `loss`: loss functions and metric functions
3. `models`: InferNet and MB-MDN
4. `my_collate_fn`: data collate functions
5. `mydataset`: dataset functions
6. `mydataset_GT`: dataset functions used for InferNet.
7. `NN_params_infer_GT2`: training for InferNet.
8. `NN_params_infer_GT3`: training for InferNet.
9. `plot`: plotting functions.
10. `training_ADAPT`: training for MBMDN in ADAPT. also can be used to train other machine learning-based method with slight modification as we did in `training_GT1`.
   - `training_GT1`: training for GT+EMB.

# Syn_Data
1. `sample_target_data_v2`: generating synthetic data
2. `sample_target_data_v2`: deciding which GT model will be used to generate the synthetic data.

