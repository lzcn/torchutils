def format_display(opt, num=1):
    """Show hierarchal information.

    Args:
        opt (dict): configuration to be displayed
        num (int): number of indent
    """
    indent = "  " * num
    string = ""
    for k, v in opt.items():
        if v is None:
            continue
        if isinstance(v, dict):
            string += "{}{} : {{\n".format(indent, k)
            string += format_display(v, num + 1)
            string += "{}}},\n".format(indent)
        elif isinstance(v, list):
            string += "{}{} : ".format(indent, k)
            one_line = ",".join(map(str, v))
            if len(one_line) < 87:
                string += "[" + one_line + "]\n"
            else:
                prefix = "  " + indent
                string += "[\n"
                for i in v:
                    string += "{}{},\n".format(prefix, i)
                string += "{}]\n".format(indent)
        else:
            string += "{}{} : {},\n".format(indent, k, v)
    return string


def update_npz(fn, results):
    if fn is None:
        return
    if os.path.exists(fn):
        pre_results = dict(np.load(fn, allow_pickle=True))
        pre_results.update(results)
        results = pre_results
    np.savez(fn, **results)


def weights_init(m):
    """
    usage: module.apply(weights_init)
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    else:
        pass


def colour(string, *args, b="", s="", c="green"):
    """Colorize a string.

    Args:
        string (str): string to colorize
        *args (Any): string = string % tuple(args)
        b (str): background color
        s (str): style,
        c (str): foreground color,

    Available formatting:
        See colorma_ for more details

        .. code-block::

            c: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
            b: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
            s: DIM, NORMAL, BRIGHT, RESET_ALL

    .. _colorma: https://pypi.org/project/colorama/
    """
    if isinstance(string, Number):
        string = str(string)
    string = string % tuple(args)
    prefix = getattr(Fore, c.upper(), "") + getattr(Back, b.upper(), "") + getattr(Style, s.upper(), "")
    suffix = Style.RESET_ALL
    return prefix + string + suffix


def get_named_class(module):
    """Get the class member in module."""
    from inspect import isclass

    return {k: v for k, v in module.__dict__.items() if isclass(v)}


def get_named_function(module):
    """Get the class member in module."""
    from inspect import isfunction

    return {k: v for k, v in module.__dict__.items() if isfunction(v)}


def one_hot(index, num):
    """Convert the LongTensor to one-hot encoding.

    Args:
        index (torch.LongTensor): index tensor
        num (int): length of one-hot encoding
    """
    index = index.view(-1, 1)
    one_hot = torch.zeros(index.numel(), num).to(index.device)
    return one_hot.scatter_(1, index, 1.0)


def get_device(gpus=None):
    """Decide which device to use for data when given gpus.

    Suppose nn.data_parallel is used for multi-gpu.
    If use multiple GPUs, then data only need to stay in CPU. If use single GPU,
    then data must move to that GPU.

    Args:
        gpus (list, optional): gpu list
    Outputs:
        - parallel: True if len(gpus) > 1
        - device: if single-gpu is used then return the gpu device, else return "cpu"
    """
    if not gpus:
        parallel = False
        device = torch.device("cpu")
        return parallel, device
    if len(gpus) > 1:
        parallel = True
        device = torch.device("cpu")
    else:
        parallel = False
        device = torch.device(gpus[0])
    return parallel, device


def to_device(data, device):
    """Move data to device.

    Args:
        data (Sequence): convert all data to given device.
        device (torch.device): target device

    Returns:
        Any: moved data
    """
    from collections import Sequence

    error_msg = "data must contain tensors or lists; found {}"
    if isinstance(data, Sequence):
        return tuple(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    raise TypeError((error_msg.format(type(data))))


def sum_of_loss(loss_dict, loss_weight):
    """Compute final loss and convert each loss to float."""
    losses = {}
    loss = 0.0
    for name, value in loss_dict.items():
        value = value.mean()
        weight = loss_weight[name]
        if weight:
            loss += value * weight
        # save the scale
        losses[name] = value.item()
    return losses, loss


def gather_loss(loss_dict, loss_weight):
    r"""Gather losses and add the 'overall' loss for backwards."""
    overall_loss = 0.0
    used_loss = []
    for name, weight in loss_weight.items():
        if weight:
            used_loss.append(name)
    losses = {}
    # only sum over the loss that has valid weight
    for name, value in loss_dict.items():
        value = value.mean()
        weight = loss_weight[name]
        if weight:
            overall_loss += value * weight
        # save the scale
        losses[name] = value
    if len(used_loss) == 1:
        losses.pop(used_loss[0])
    losses["overall"] = overall_loss
    return losses


def gather_accuracy(accuracy):
    return {k: v.sum().item() / v.numel() for k, v in accuracy.items()}


def gather_tensor(tensor):
    r"""Gather mean value of tensors.

    Compute the averaged value :meth:`\frac{1}{n}\sum_i v_i, v \in\matchbb{R}^n`.

    Args:
        tensor (dict): list of tensors.

    Returns:
        dict: averaged results
    """
    return {k: v.sum().item() / v.numel() for k, v in tensor.items()}


def load_pretrained_lossly(net, pretrained):
    """Load weights lossly.

    Load weights that match the the model. Unloaded weighted will be logged.

    Types of unloaded weights:
        - Missing keys: weights not in pretrained state
        - Unexpected keys: weights not in given net
        - Unmatched keys: shape mismatched weights
    Args:
        net (torch.nn.Moudle): model
        pretrained (str): path to pre-traiend model

    """
    # load weights from pre-trained model lossly
    num_devices = torch.cuda.device_count()
    map_location = {"cuda:{}".format(i): "cpu" for i in range(num_devices)}
    LOGGER.info("Loading pre-trained model from %s.", pretrained)
    state_dict = torch.load(pretrained, map_location=map_location)
    net_param = net.state_dict()
    unmatched_keys = []
    for name, param in state_dict.items():
        if name in net_param and param.shape != net_param[name].shape:
            unmatched_keys.append(name)
    for name in unmatched_keys:
        state_dict.pop(name)
    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
    missing_keys = list(set(missing_keys) - set(unmatched_keys))
    LOGGER.info("Missing keys: %s", ", ".join(missing_keys))
    LOGGER.info("Unexpected keys: %s", ", ".join(unexpected_keys))
    LOGGER.info("Unmatched keys: %s", ", ".join(unmatched_keys))
    return net


def init_optimizer(net, optim_param):
    """Init Optimizer given OptimParam instance and net."""
    # Optimizer and LR policy class
    grad_class = get_named_class(torch.optim)[optim_param.name]
    lr_class = get_named_class(torch.optim.lr_scheduler)[optim_param.lr_scheduler]
    # Optimizer LR policy configurations
    grad_param = optim_param.grad_param
    lr_param = optim_param.scheduler_param
    # sub-module specific configuration
    named_groups = optim_param.groups
    if named_groups:
        param_groups = []
        for name, gropus in named_groups.items():
            sub_module = operator.attrgetter(name)(net)
            param_group = dict(params=sub_module.parameters(), **gropus)
            param_groups.append(param_group)
    else:
        param_group = net.parameters()
    # get instances
    optimizer = grad_class(param_groups, **grad_param)
    lr_scheduler = lr_class(optimizer, **lr_param)
    return optimizer, lr_scheduler
