# All plot functions are here.

def plot_net(writer,model,input):
    """
    In tensorboard, visualize the model structure
    Args:
        writer:
        model:
        input:
    """
    writer.add_graph(model, input_to_model=input, verbose=False)

# def plot_conv(writer,model,epoch):
#     """
#     In tensorboard, visualize the conv weight as a picture
#     Args:
#         writer:
#         model:
#         epoch:
#     """
#     for name,param in model.named_parameters():
#         if 'conv' in name and 'weight' in name:
#             in_channels = param.size()[1]
#             out_channels = param.size()[0]
#             k_w, k_h = param.size()[3], param.size()[2]
#             kernel_all = param.view(-1, 1, k_w, k_h)
#             kernel_grid = torchvision.utils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=in_channels)
#             win_str = f'{name}_all'+str(epoch)
#             # viz.images(kernel_all, win=win_str,env="001",opts=dict(title = win_str))
#             # writer.add_image(f'{name}_all', kernel_grid, global_step=epoch)


def plot_conv_weight(writer,mlp,epoch,tag_str):
    """
    In tensorboard, visualize the conv weight.
    Args:
        mlp:
        epoch:
    """
    for name,param in mlp.named_parameters():
        if 'conv' in name and 'weight' in name:
            writer.add_histogram(tag = name+tag_str, values=param.data.clone().cpu().numpy(),global_step=epoch)


def plot_mu_weight(writer,mlp,epoch,tag_str):
    """
    In tensorboard, visualize the 'mu' weight
    Args:
        mlp:
        epoch:
    """
    for name, param in mlp.named_parameters():
        if 'z_mu' in name and 'weight' in name:
            writer.add_histogram(tag=name + tag_str, values=param.data.clone().cpu().numpy(), global_step=epoch)


def plot_pi_weight(writer,mlp,epoch,tag_str):
    """
    In tensorboard, visualize the 'pi' weight
    Args:
        mlp:
        epoch:
    """
    for name, param in mlp.named_parameters():
        if 'z_pi' in name and 'weight' in name:
            writer.add_histogram(tag=name + tag_str, values=param.data.clone().cpu().numpy(), global_step=epoch)


def plot_sigma_weight(writer,mlp,epoch,tag_str):
    """
    In tensorboard, visualize the 'sigma' weight
    Args:
        mlp:
        epoch:
    """
    for name, param in mlp.named_parameters():
        if 'z_sigma' in name and 'weight' in name:
            writer.add_histogram(tag=name + tag_str, values=param.data.clone().cpu().numpy(), global_step=epoch)


def plot_alpha(writer,mlp,epoch,tag_str):
    """
    In tensorboard, visualize the 'alpha weight
    Args:
        writer:  tensorboard writer
        mlp:     the NN model
        epoch:   epoch
        tag_str: the name of the plot window
    """
    for name, param in mlp.named_parameters():
        if 'block_alpha' in name and 'weight' in name:
            writer.add_histogram(tag=name + tag_str, values=param.data.clone().cpu().numpy(), global_step=epoch)


def plot_labda(writer,mlp,epoch,tag_str):
    '''
    In tensorboard, visualize the 'labda' weight

    :param writer:
    :param mlp:
    :param epoch:
    :param tag_str:
    :return:
    '''
    for name, param in mlp.named_parameters():
        if 'block_labda' in name and 'weight' in name:
            writer.add_histogram(tag=name + tag_str, values=param.data.clone().cpu().numpy(), global_step=epoch)


win_train_loss_str = "The Loss of BATCH in the Training Data"
win_vali_loss_str = "The Loss in the Vali Data"
win_train_epoch_loss_str = "The Loss of EPOCH in the Training Data"

def draw_loss(viz, X_step, loss, win_str):
    '''
    In visdom, visualize the training loss
    :param viz:
    :param X_step:
    :param loss:
    :param win_str:
    :return:
    '''
    viz.line(X = [X_step], Y = [loss],win=win_str, update="append",
        opts= dict(title=win_str))


win_vali_metric_str = "NLL in vali data"
def draw_metric(viz,X_step, total_vali_metric, GT_metric, win_str, MODEL_NAME = 'pred'):
    '''
    In visdom, visualize the vali metrics

    :param viz:
    :param X_step:
    :param total_vali_metric:
    :param GT_metric:
    :param win_str:
    :param MODEL_NAME:
    :return:
    '''
    legend_str = [MODEL_NAME] + ['GT-1','GT-2(NN)']
    viz.line(X = [X_step], Y = [[total_vali_metric,GT_metric[0,0],GT_metric[0,-1]]],win=win_str, update="append", opts= dict(title=win_vali_metric_str, legend=legend_str, showlegend=True,xlabel="epoch", ylabel="NLL"))