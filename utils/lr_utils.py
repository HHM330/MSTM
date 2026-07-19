import numpy as np
import math
import omegaconf

def to_list(inp):
    if inp is None:
        return None
    if isinstance(inp, omegaconf.listconfig.ListConfig):
        inp = omegaconf.OmegaConf.to_container(inp, resolve=True)
    return inp

def lr_tuner(lr_init, optimizer, epoch_size, tune_dict, global_step=1, use_warmup=False, warmup_epochs=1, lr_min=1e-6):
    if use_warmup and global_step <= epoch_size * warmup_epochs:
        if global_step == 1:
            print(">>> Using warmup strategy!")
        lr = global_step / epoch_size * lr_init
    else:
        if use_warmup:
            new_step = global_step - epoch_size
        else:
            new_step = global_step

        tune_dict = dict(tune_dict)
        decay_strategy_name = tune_dict['name']

        if decay_strategy_name == "piecewise":
            decay_steps = to_list(tune_dict.get("decay_steps"))
            decay_epochs = to_list(tune_dict.get("decay_epochs"))
            decay_rates = to_list(tune_dict.get("decay_rates"))

            if decay_steps and decay_epochs:
                raise ValueError(
                "decay_steps and decay_epochs in tune_dict of lr_tuner are both set, only one of them can be set"
                )
            if decay_epochs:
                decay_steps = [epoch_size * x for x in decay_epochs]
            
            decay_cnt = np.sum(new_step > np.asarray(decay_steps))
            if decay_cnt == 0:
                decay_mult = 1.0
            else:
                decay_mult = decay_rates[decay_cnt - 1]
            
            lr = max(lr_init * decay_mult, lr_min)

        elif decay_strategy_name == "exponential":
            decay_step = tune_dict.get("decay_step")
            decay_epoch = tune_dict.get("decay_epoch")
            decay_rate = tune_dict.get("decay_rate")
            staircase = tune_dict.get("staircase")

            if decay_step and decay_epoch:
                raise ValueError(
                "decay_step and decay_epoch in tune_dict of lr_tuner are both set, only one of them can be set"
                )
            if decay_epoch:
                decay_step = epoch_size * decay_epoch
            
            decay_index = new_step // decay_step if staircase else new_step / decay_step
            lr = max(lr_init * math.pow(decay_rate, decay_index), lr_min)
        elif decay_strategy_name == "cosine":
            T_max = tune_dict.get("T_max")
            warmup_epochs = tune_dict.get("warmup_epochs", 0)
            min_lr = tune_dict.get("min_lr", 1e-6)

            if T_max is None:
                raise ValueError("cosine scheduler requires T_max")

            if use_warmup and global_step <= epoch_size * warmup_epochs:
                lr = global_step / (epoch_size * warmup_epochs) * lr_init
            else:
                current_step = global_step - epoch_size * warmup_epochs
                total_steps = epoch_size * T_max
                lr = min_lr + (lr_init - min_lr) * \
                     (1 + math.cos(math.pi * current_step / total_steps)) / 2

            lr = max(lr, min_lr)
        else:
            raise NotImplementedError("decay_strategy_name {} is not supported".format(decay_strategy_name))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
