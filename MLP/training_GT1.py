#  This is training for GT1+EMD
## Training process of GT2+EMB is similarly.

from common import *
from mydataset import *
from models import Conv_1_4
reload(loss)
reload(plot)
reload(my_collate_fn)

from MLP.Config.config_MDN import DefaultConfig
opt = DefaultConfig()

######*********CHECK CAREFULLY***********########
MODEL_NAME = "GT1+EMB"
q=opt.q
lr_for_pi = opt.lr_for_pi
lr_for_shape = opt.lr_for_shape
lr_for_scale = opt.lr_for_scale
learning_rate = opt.learning_rate

MB_MDN_Flag = opt.Conv_1_4

batch_size = opt.batch_size
seed = opt.seed
EPOCH_NUM = opt.EPOCH_NUM
######*********CHECK CAREFULLY***********########
def get_model_save_path(flag,seed):
    model_params_MLP = ""
    if flag:
        model_params_MLP = opt.net_root_path + MODEL_NAME + "_noise=" + str(opt.noise_pct) + "_seed=" + str(
            seed) + ".pth"
    else:
        model_params_MLP = opt.net_root_path + MODEL_NAME + str(seed) + ".pth"

    return model_params_MLP

def get_data_loader(dataset, shuffled_indices, opt):
    """
    To get dataloader according to shuffled 'shuffled_indices'
    Args:
        dataset:
        shuffled_indices:
        opt:

    Returns:

    """
    train_idx,val_idx,test_idx = get_data_idx(shuffled_indices,opt)
    my_collate_fn = functools.partial(my_collate_fn_1GT, GT_CHOSEN=opt.GT_1)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(train_idx), collate_fn=my_collate_fn)
    val_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx), collate_fn=my_collate_fn)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(test_idx), collate_fn=my_collate_fn)

    return train_loader,val_loader,test_loader

def get_params(mlp,opt):
    """
    Set learning rates for different layers and return params for training
    Args:
        mlp:
        opt:

    Returns: params

    """
    shape_params = list(map(id, mlp.z_shape.parameters()))
    scale_params = list(map(id, mlp.z_scale.parameters()))
    pi_params = list(map(id, mlp.z_pi.parameters()))

    params_id = shape_params + scale_params + pi_params

    base_params = filter(lambda p: id(p) not in params_id, mlp.parameters())
    params = [{'params': base_params},
            {'params': mlp.z_pi.parameters(), 'lr': lr_for_pi},
            {'params': mlp.z_shape.parameters(), 'lr': lr_for_shape},
            {'params': mlp.z_scale.parameters(), 'lr': lr_for_scale}]

    return params

def trainer(train_loader, val_loader, test_loader, mlp, opt, device):
    """
    Main body of a training process. Called by objective function
    Args:
        train_loader:
        val_loader:
        mlp:
        opt:    params to be held
        device:

    Returns: NLL in validation set

    """
    params = get_params(mlp,opt)
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.StepLR_step_size, gamma=opt.StepLR_gamma)

    total_train_step = 0

    for epoch in range(EPOCH_NUM):
        mlp.train()
        epoch_train_loss = 0

        for batch_id, data in enumerate(train_loader):
            input_data, _, target_loss, setting, _ = data
            # Do the inference
            input_data = input_data.to(device)
            target_loss = target_loss.to(device)
            pi, mu, sigma = mlp(input_data)

            loss = loss_fn_wei(pi, mu, sigma, target_loss, opt.N_gaussians, opt.TARGET, opt.SAFETY, device)
            epoch_train_loss += loss.detach().cpu()

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(mlp.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            total_train_step += 1
        scheduler.step()

        if opt.DRAW_VAL or opt.PRINT_VAL:
            if epoch % 2 == 0:
                mlp.eval()
                with torch.no_grad():
                    total_vali_metric, GT_metric, odds = validate(mlp, val_loader, opt.N_gaussians, opt.MIN_LOSS, device, detail=True)
                if opt.DRAW_VAL:
                    draw_metric(viz, epoch, total_vali_metric.detach().cpu(), GT_metric, plot.win_vali_metric_str, MODEL_NAME)
                    writer.add_scalars(opt.vali_main_tag_str, {MODEL_NAME: total_vali_metric.detach().cpu(),
                                                            "GT-1": GT_metric[0, 0],
                                                            "GT-2": GT_metric[0, -1]}, epoch+1)
                if opt.PRINT_VAL:
                    print(f"vali = {total_vali_metric}, GT = {GT_metric}")
                mlp.train()

        model_params_MLP = get_model_save_path(opt.ARTIFICIAL, seed)
        torch.save(mlp.state_dict(), model_params_MLP)

        print(f"epoch={epoch}, loss = {epoch_train_loss.detach().cpu()}")

    mlp.eval()
    with torch.no_grad():
        total_vali_metric, GT_metric, odds = validate(mlp, val_loader, opt.N_gaussians, opt.MIN_LOSS, device, detail=True)
        GT_metric_1 = GT_metric[0, 0]

    metric_diff = GT_metric_1 - total_vali_metric
    print(f"GT_metric = {GT_metric}, NN prediction = {total_vali_metric}")
    print(f"the GTs: GT1,GT2(common),GT2(SA), GT2(NN)")

    mlp_all.append(mlp.state_dict())

    return metric_diff,total_vali_metric,GT_metric

ans = []
pred_metric = []
mlp_all = []

if __name__ == '__main__':

    times = [1]

    setup_seed(seed)

    viz = visdom.Visdom(env=opt.env_str)
    writer = SummaryWriter(log_dir="logs-MLP/" + opt.logs_str, flush_secs=60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = myDataset(opt.train_path, opt.target_path_metric, opt.target_path_loss, opt.data_key_path, opt.NLL_metric_path)
    shuffled_indices = save_data_idx(dataset, opt)
    train_loader, val_loader, test_loader = get_data_loader(dataset, shuffled_indices, opt)

    for i in times:

        model = Conv_1_4(opt.N_gaussians).to(device)

        # Get initial NLL in vali set.
        if opt.DRAW_VAL:
            model.eval()
            with torch.no_grad():
                total_vali_metric, GT_metric, odds = validate(model, val_loader, opt.N_gaussians, opt.MIN_LOSS, device, detail=True)
            writer.add_scalars(opt.vali_main_tag_str, {MODEL_NAME: total_vali_metric.detach().cpu()}, 0)
            model.train()

        # Get final NLL in vali set.
        performance = trainer(train_loader, val_loader, test_loader, model, opt, device)

        ans.append(performance[0].cpu().numpy())
        pred_metric.append(performance[1].cpu().numpy())

        print(f"running time = {i}, performance = {performance[0]}")

    print("-----mean: " , np.mean(ans))
    print("-----mean metric: " , np.mean(pred_metric))
