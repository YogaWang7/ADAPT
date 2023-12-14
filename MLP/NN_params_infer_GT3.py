
import torch.utils.data
import mydataset_GT
from functorch import vmap
from mydataset_GT import myDataset
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F

from visdom import Visdom

import time
from GT_model.GT_3 import SA_for_PT_funcs_delta_eq1
from importlib import reload

from torch.cuda.amp import GradScaler,autocast
import pickle

scaler = GradScaler(backoff_factor = 0.1)
import scipy
from torch.autograd.gradcheck import gradcheck

from Config import config_GT3
import loss
import plot
import my_collate_fn
from my_collate_fn import my_collate_fn_GT2
from utils import *
from MLP.models import MLP_GT3_2


# torch.set_default_tensor_type(torch.FloatTensor)

reload(config_GT3)
reload(loss)
reload(plot)
reload(mydataset_GT2)
reload(SA_for_PT_funcs_delta_eq1)
reload(my_collate_fn)

win_train_loss_str = "The Loss of BATCH in the Training Data"
win_vali_loss_str = "The Loss in the Vali Data"
win_train_epoch_loss_str = "The Loss of EPOCH in the Training Data"
win_vali_metric_str = "The NLL of ALL vali data"
def draw_loss(X_step, loss, win_str):
    viz.line(X = [X_step], Y = [loss],win=win_str, update="append",
        opts= dict(title=win_str))

def draw_metric(X_step, total_vali_metric):

    viz.line(X = [X_step], Y = [total_vali_metric],win=win_vali_metric_str, update="append", opts= dict(title=win_vali_metric_str, legend=['pred','GT-1','GT-2'], showlegend=True,xlabel="epoch", ylabel="NLL"))

from plot import plot_alpha
from plot import plot_labda

opt = config_GT3.DefaultConfig()

total_train_step = 0
total_test_step = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bound_alpha = torch.tensor([-0.3,0.3],device=device)
bound_labda = torch.tensor([0.01,18],device=device)

def get_model_save_path(flag,seed):
    model_params_MLP = ""
    if flag:
        model_params_MLP = opt.net_root_path + "NN_params_infer_GT3_artificial_v2_" + "noise=" + str(opt.noise_pct) + "_seed=" + str(
            seed) + ".pth"
    else:
        model_params_MLP = opt.net_root_path + "NN_params_infer_GT3_seed=" + str(seed) + ".pth"

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
    train_idx, val_idx, test_idx = get_data_idx(shuffled_indices,opt)
    # train_idx,val_idx,test_idx = get_data_idx_bidfee(shuffled_indices,opt)
    # train_idx,val_idx,test_idx = get_data_idx_bidinc(shuffled_indices,opt)

    train_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(train_idx), collate_fn=my_collate_fn_GT2)
    val_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx), collate_fn=my_collate_fn_GT2)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(test_idx), collate_fn=my_collate_fn_GT2)

    return train_loader,val_loader,test_loader

def get_params(mlp,opt):
    """
    Set learning rates for different layers and return params for training
    Args:
        mlp:
        opt:

    Returns: params

    """
    alpha_params = list(map(id, mlp.block_alpha.parameters()))

    params_id = alpha_params
    other_params = filter(lambda p: id(p) not in params_id, mlp.parameters())
    params = [{'params': other_params},
                {'params': mlp.block_alpha.parameters(), 'lr': opt.lr_for_alpha}]

    return params

def f_ts(x, alpha, device):
    """
    utility function = f = 1-f_2

    Args:
        x:
        alpha:
        device:

    Returns: y2= 1-log(-alpha*x), and restrict the return value within a range.

    """
    # y1 = torch.clamp(-alpha*x,max=torch.tensor(85.,device=device))#.to(device)       # torch.exp(x), x<85
    y2 = (1-torch.exp(torch.clamp( -alpha*x, max=torch.tensor(85.,device=device))))  #.to(device)

    return y2


def f_Equi_ts_log_vmap(t, v, d, b, alpha):
    root = torch.log( torch.abs( (f_ts(((v-d*(t-1) - b)),alpha,device)) )) - torch.log(
        torch.abs((f_ts(((v-d*(t-1) - b)),alpha,device)) - f_ts(-b,alpha,device)))

    return root

def get_LEN_T_ts(v,b,d, max_T):
    """
    Get the length of this auction
    Args:
        v:
        b:
        d:
        max_T:

    Returns:
        LEN: Length of U vector
        T: max possible duration: max_T or theoratic T

    """

    if d == 0:          # fixed-price
        # T = np.inf
        T = max_T
    else:               # asc-price
        T = torch.floor((v - b) / d)

    LEN = max_T
    return LEN,T

def get_U_GT3_ts_log_vmap(LEN, T, v, d, b, alpha, eps = 1e-30, device = device):

    # U: the prob. that someone offers a bid in t_th round
    # log(1) = 0.
    U_head = torch.tensor([0.,0.],requires_grad=True,device=device)

    idx = torch.arange(2,LEN+1).to(device)

    # Vectorize this calculation
    v_extend = torch.expand_copy(v,size=idx.shape).to(device)
    b_extend = torch.expand_copy(b,size=idx.shape).to(device)
    d_extend = torch.expand_copy(d,size=idx.shape).to(device)
    alpha_extend = torch.expand_copy(alpha,size=idx.shape).to(device)

    f_Equi_vec = vmap(f_Equi_ts_log_vmap)  # [N, D], [N, D] -> [N]
    U_root = f_Equi_vec(idx, v_extend, d_extend, b_extend, alpha_extend)    # U_root is log-value

    U_T_1 = torch.log(torch.tensor([eps],requires_grad=True,device=device))
    U_i_log = torch.concat([U_head,U_root,U_T_1])

    U_i_log = U_i_log.to(device)

    del idx
    del v_extend
    del b_extend
    del d_extend
    return U_i_log


def loss_fn_3(input_data, Alpha, Target_data, eps, device):
    '''
    Loss function for infernet.

    Args:
        input_data:
        Alpha:
        Target_data:
        eps:
        device:

    Returns:

    '''
    # Note: P[i][t] = U[i][1]*U[i][2]*...*(1-U[i][t+1])
    loss_sum = torch.tensor(0.,device=device,requires_grad=True)
    for i in range(len(input_data)):

        # Get target data
        target = Target_data[i, :].long()

        # Solve for U from Equi. condt.
        # Get settings
        setting = input_data[i, :, 0:3]
        d = setting[0, 0]
        b = setting[0, 1]
        v = setting[0, 2]

        # 1.Cal LEN,T
        LEN,T = get_LEN_T_ts(v,b,d,max(target))
        # 2.Select target data and drop those duration values larger than T

        idx = torch.nonzero(target)
        target_nonzero = target[idx]  #.reshape(1,-1).squeeze().long()      # int

        U_log = get_U_GT3_ts_log_vmap(LEN, T, v, d, b, Alpha[i,:],eps = eps, device=device)
        # assert not (torch.any(torch.isnan(U_log.detach()))), f"U_log has NaN and U_log = {U_log.detach()}"

        U_log_cumsum = torch.cumsum(U_log,dim=0)#.to(device)
        # assert U_log_cumsum.shape == U.shape, "U_log_cumsum.shape != U.shape"

        U_log_cumsum_extend = torch.repeat_interleave(U_log_cumsum[None,:], repeats=len(target_nonzero),dim=0)#.to(device)
        del U_log_cumsum

        U_sum_1_idx = target_nonzero
        U_sum_2_idx = target_nonzero.squeeze()+1
        U_sum_1 = torch.gather(U_log_cumsum_extend,dim=1,index=U_sum_1_idx)#.to(device)
        del U_log_cumsum_extend
        del U_sum_1_idx

        U_sum_2 = torch.log(torch.max(1-torch.exp(torch.gather(U_log,0,index=U_sum_2_idx)),torch.tensor(opt.MIN_LOSS)))#.to(device)

        del U_log
        del U_sum_2_idx
        loss_sum = U_sum_1.sum() + U_sum_2.sum() + loss_sum
        del U_sum_1

    return -loss_sum/len(input_data)


def validate_params(mlp, test_loader, eps, device):
    '''
    Metric Calculation

    Args:
        mlp:
        test_loader:
        eps:
        device:

    Returns:

    '''

    loss_sum = torch.tensor(0., device=device, requires_grad=False)
    GT_metric = torch.tensor([0.,0.,0.]).reshape(1,-1)

    cnt = 0
    for batch_id, data in enumerate(test_loader):

        input_data, target_metric,_, target_params_data, _, metric_data= data

        # print(f"---- {batch_id} batch----")
        # Do the inference
        input_data = input_data.to(device)
        target_data = target_metric.to(device)
        # target_params_data = target_params_data.to(device)      # params inferred by SA

        Alpha= mlp(input_data)
        Alpha = Alpha.detach().cpu().numpy()

        GT_metric += torch.sum(metric_data,dim=0)
        cnt += len(input_data)

        for i in range(len(Alpha)):

            # Get target data
            target = target_data[i, :]
            idx = torch.nonzero(target)
            target_nonzero = target[idx].detach().cpu().squeeze().numpy()
            target_ls = [int(x) for x in target_nonzero]

            # Solve for U from Equi. condt.
            # Get settings
            setting = input_data[i, :, 0:3]
            d = setting[0, 0].detach().cpu().numpy().item()
            b = setting[0, 1].detach().cpu().numpy().item()
            v = setting[0, 2].detach().cpu().numpy().item()

            LEN,T =  SA_for_PT_funcs_delta_eq1.get_LEN_T(v,b,d,max(target_ls))

            # Solve for U
            U = SA_for_PT_funcs_delta_eq1.get_U_GT3(LEN,v,d,b,Alpha[i].item(),eps=0.)

            nll_metric = SA_for_PT_funcs_delta_eq1.get_nll_meric(target_ls, U, LEN,TARGET = 1)

            loss_sum = nll_metric + loss_sum

    return loss_sum / cnt, GT_metric/ cnt

def trainer(train_loader, val_loader, test_loader, mlp, opt, device):
    """
    Main body of a training process. Called by objective function
    Args:
        train_loader:
        val_loader:
        mlp:
        opt:    params to be held
        device:

    Returns: performance(avg NLL in last 5 epoch) in validation set

    """
    params = get_params(mlp,opt)
    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.Exp_gamma, last_epoch=-1)

    total_train_step = 0
    EPOCH_NUM = opt.EPOCH_NUM

    for epoch in range(EPOCH_NUM):
        mlp.train()
        epoch_train_loss = 0

        for batch_id, data in enumerate(train_loader):
            input_data, _, target_loss, _, _, _ = data
            # Do the inference
            input_data = input_data.to(device)
            target_loss = target_loss.to(device)

            # with autocast():
            alpha = mlp(input_data)

            # Cal the MLE loss and draw the distrb.
            loss = loss_fn_3(input_data, alpha, target_loss, opt.MIN_LOSS, device)

            epoch_train_loss += loss.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            draw_loss(total_train_step, loss.detach().cpu(), win_train_loss_str)

            total_train_step += 1

            torch.cuda.empty_cache()

        scheduler.step()
        print(f"========== IN EPOCH {epoch} the total loss is {epoch_train_loss} ==========")

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

    # Save the net
    model_params_MLP = get_model_save_path(opt.ARTIFICIAL,seed)
    torch.save(mlp.state_dict(), model_params_MLP)

    mlp.eval()
    with torch.no_grad():
        # The validation
        total_vali_metric, GT_metric, _ = validate(mlp, val_loader, opt.N_gaussians, opt.MIN_LOSS, device,detail=True)
    mlp.train()

    return total_test_metric,GT_metric


if __name__ == '__main__':

    seed = opt.seed
    setup_seed(seed)
    running_times=[1]

    timestamp = int(time.time())
    time_str = str("_") + time.strftime('%y%m%d%H%M%S', time.localtime(timestamp))
    print(f"time_str = {time_str}")
    env_str = "params_NLL_infer_seed=" + str(seed) + time_str
    viz = Visdom(env=env_str)

    viz.line(X=[0.], Y=[0.], env=env_str, win=win_train_loss_str, opts=dict(title=win_train_loss_str))
    viz.line(X=[0.], Y=[0.], env=env_str, win=win_train_epoch_loss_str, opts=dict(title=win_train_epoch_loss_str))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = myDataset(opt.train_path, opt.target_path_metric, opt.target_path_loss, opt.params_opitim_path,
                        opt.data_key_path, opt.NLL_metric_path)
    shuffled_indices = save_data_idx(dataset, opt)
    train_loader,val_loader,test_loader = get_data_loader(dataset, shuffled_indices, opt)

    for i in running_times:

        model = MLP_GT3_2().to(device)

        total_test_metric,GT_metric = trainer(train_loader, val_loader, test_loader, model, opt, device)
        print(f"========== IN Vali dataset, NN Model:  {total_test_metric.detach().cpu().numpy()} ==========")
        print(f"========== IN Vali dataset, the GTs: {GT_metric.detach().cpu().numpy()} ==========")
        print(f"========== IN Vali dataset, the GTs: GT1,GT2(common),GT2(SA) ==========")
