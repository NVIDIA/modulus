import torch


def instantiate_scheduler(optimizer, config):
    if config.opt_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.opt_scheduler_T_max
        )
    elif config.opt_scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.opt_step_size, gamma=config.opt_gamma
        )
    else:
        raise ValueError(f"Got {config.opt.scheduler=}")
    return scheduler
