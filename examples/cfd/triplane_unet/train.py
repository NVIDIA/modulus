import sys

import matplotlib

matplotlib.use("Agg")  # Set the backend to Agg

import os
from contextlib import suppress
from pathlib import Path
from timeit import default_timer
from typing import List, Tuple, Union

import torch
import yaml
from torch.cuda import amp

from config_parser import load_config, parse_args
from src.data import instantiate_datamodule
from src.losses import get_loss
from src.networks import instantiate_network
from src.optim.schedulers import instantiate_scheduler
from src.utils.average_meter import AverageMeter, AverageMeterDict, Timer
from src.utils.loggers import init_logger
from src.utils.seed import set_seed


def save_config(config: dict):
    with open(os.path.join(config.output, "config.yml"), "w") as f:
        yaml.dump(dict(config), f)

    if "wandb" in config.logger_types:
        import wandb

        wandb.config.update(config)
        wandb.save(os.path.join(config.output, "config.yml"), policy="now")


def _delete_previous_checkpoints(config):
    checkpoints_to_delete = []
    for f in os.listdir(config.output):
        if f.startswith("model_") and f.endswith(".pth"):
            checkpoints_to_delete.append(f)
    checkpoints_to_delete.sort()
    checkpoints_to_delete = checkpoints_to_delete[: -config.num_checkpoints]
    print(f"Deleting {len(checkpoints_to_delete)} checkpoints")
    for f in checkpoints_to_delete:
        try:
            os.remove(os.path.join(config.output, f))
        except FileNotFoundError:
            pass


def save_state(model, optimizer, scheduler, epoch, tot_iter, config):
    save_path = os.path.join(config.output, f"model_{epoch:05d}.pth")
    print(f"Saving model at epoch {epoch} to {save_path}")
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "tot_iter": tot_iter,
    }
    # Save the file with 0000X format
    torch.save(state_dict, save_path)
    _delete_previous_checkpoints(config)


def get_autocast(precision: str = None):
    if precision == "amp":
        print("Using amp autocast")
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        print("Using amp autocast with bfloat16")
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


@torch.no_grad()
def eval(model, datamodule, config, loss_fn=None):
    model.eval()
    test_loader = datamodule.test_dataloader(batch_size=config.batch_size, num_workers=0)
    eval_meter = AverageMeterDict()
    visualize_data_dicts = []
    eval_timer = Timer()
    for i, data_dict in enumerate(test_loader):
        eval_timer.tic()
        out_dict = model.eval_dict(data_dict, loss_fn=loss_fn, datamodule=datamodule)
        out_dict["inference_time"] = eval_timer.toc()
        eval_meter.update(out_dict)
        if i % config.test_plot_interval == 0:
            visualize_data_dicts.append(data_dict)
        if i % config.test_print_interval == 0:
            # Print eval dict
            print(f"Eval {i}: {eval_meter.avg}")

    # Merge all dictionaries
    merged_image_dict = {}
    merged_point_cloud_dict = {}
    if hasattr(model, "image_pointcloud_dict"):
        for i, data_dict in enumerate(visualize_data_dicts):
            image_dict, pointcloud_dict = model.image_pointcloud_dict(
                data_dict, datamodule=datamodule
            )
            for k, v in image_dict.items():
                merged_image_dict[f"{k}_{i}"] = v
            for k, v in pointcloud_dict.items():
                merged_point_cloud_dict[f"{k}_{i}"] = v
    elif hasattr(model, "image_dict"):
        for i, data_dict in enumerate(visualize_data_dicts):
            image_dict, pointcloud_dict = model.image_dict(data_dict, datamodule=datamodule)
            for k, v in image_dict.items():
                merged_image_dict[f"{k}_{i}"] = v

    model.train()
    return eval_meter.avg, merged_image_dict, merged_point_cloud_dict


def train(config, device: Union[torch.device, str] = "cuda:0"):
    # Initialize the device
    if isinstance(device, str):
        device = torch.device(device)

    # Initialize the model
    model = instantiate_network(config)
    # if model.device == torch.device("cpu"):
    model = model.to(device)

    # Initialize the dataloaders
    datamodule = instantiate_datamodule(config)
    train_loader = datamodule.train_dataloader(batch_size=config.batch_size, shuffle=True)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = instantiate_scheduler(optimizer, config)
    # Initialize the loss function
    loss_fn = get_loss(config.loss_name)
    if config.eval_loss_name is None:
        eval_loss_fn = loss_fn
    else:
        eval_loss_fn = get_loss(config.eval_loss_name)
    scaler = amp.GradScaler()

    # AMP precision
    autocast = get_autocast(config.amp_precision)

    # If time_limit is set, break the training loop before the time limit
    average_epoch_time = 0
    if config.time_limit is not None:
        # time limit is the form hh:mm:ss
        time_limit = sum(
            int(x) * 60**i for i, x in enumerate(reversed(config.time_limit.split(":")))
        )
        print(f"Time limit: {time_limit} seconds")
        start_time = default_timer()
    else:
        time_limit = None

    # Resume if resume is True
    start_epoch = 0
    tot_iter = 0
    if config.resume and os.path.exists(config.output):
        print(f"Resuming from {config.output}")
        # Find the latest checkpoint
        checkpoints = []
        for f in os.listdir(config.output):
            if f.startswith("model_") and f.endswith(".pth"):
                checkpoints.append(f)
        checkpoints.sort()
        # Load if there is a checkpoint
        if len(checkpoints) == 0:
            print("No checkpoints found")
        else:
            print(f"Found {len(checkpoints)} checkpoints")
            print(f"Loading {checkpoints[-1]}")
            checkpoint_path = os.path.join(config.output, checkpoints[-1])
            checkpoint = torch.load(checkpoint_path)

            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            tot_iter = checkpoint["tot_iter"]

    # Initialize the logger first to access experiment status
    loggers = init_logger(config)

    # N_sample = 1000
    save_config(config)
    for ep in range(start_epoch, config.num_epochs):
        model.train()
        t1 = default_timer()
        train_l2_meter = AverageMeter()

        for data_dict in train_loader:
            optimizer.zero_grad()

            with autocast():
                loss_dict = model.loss_dict(data_dict, loss_fn=loss_fn, datamodule=datamodule)

            loss = 0
            for k, v in loss_dict.items():
                weight_name = k + "_weight"
                if hasattr(config, weight_name) and getattr(config, weight_name) is not None:
                    v = v * getattr(config, weight_name)
                loss = loss + v.mean()

            # Assert loss is valid
            assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"

            if config.amp_precision:
                scaler.scale(loss).backward()

                if config.amp_clip_grad:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.amp_grad_max_norm)

                    # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                    scaler.step(optimizer)

                    # Updates the scale for next iteration.
                    scaler.update()
            else:
                loss.backward()
            optimizer.step()

            train_l2_meter.update(loss.item())
            loggers.log_scalar("train/iter_lr", scheduler.get_lr()[0], tot_iter)
            loggers.log_scalar("train/iter_loss", loss.item(), tot_iter)
            for k, v in loss_dict.items():
                loggers.log_scalar(f"train/{k}", v.item(), tot_iter)
            if tot_iter % config.train_print_interval == 0:
                print(f"Iter {tot_iter} loss: {loss.item():.4f}")

            tot_iter += 1
            torch.cuda.empty_cache()

        scheduler.step()
        t2 = default_timer()
        print(f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2 loss: {train_l2_meter.avg:.4f}")
        loggers.log_scalar("train/epoch_train_l2", train_l2_meter.avg, tot_iter)
        loggers.log_scalar("train/train_epoch_duration", t2 - t1, tot_iter)

        if ep % config.eval_interval == 0 or ep == config.num_epochs - 1:
            eval_dict, eval_images, eval_point_clouds = eval(
                model, datamodule, config, eval_loss_fn
            )
            for k, v in eval_dict.items():
                print(f"Epoch: {ep} {k}: {v:.4f}")
                loggers.log_scalar(f"eval/{k}", v, tot_iter)
            for k, v in eval_images.items():
                loggers.log_image(f"eval_vis/{k}", v, tot_iter)
            if config.log_pointcloud:
                for k, v in eval_point_clouds.items():
                    loggers.log_pointcloud(f"eval_vis/{k}", v[..., :3], v[..., 3:], tot_iter)

        # Save the weights, optimization state, and scheduler state into one file
        if ep % config.save_interval == 0:
            # save the model
            save_state(model, optimizer, scheduler, ep, tot_iter, config)

        # Update average epoch time
        if average_epoch_time == 0:
            average_epoch_time = t2 - t1
        else:
            average_epoch_time = (t2 - t1) * 0.1 + average_epoch_time * 0.9

        # Break the training loop if the time limit is reached
        if time_limit is not None:
            if default_timer() - start_time + 2 * average_epoch_time > time_limit:
                print(f"Time limit {time_limit} seconds reached. Breaking the training loop.")
                # Save model
                save_state(model, optimizer, scheduler, ep, tot_iter, config)
                # Write the status to the output directory to status.txt file
                with open(os.path.join(config.output, "status.txt"), "w") as f:
                    f.write("resuming")
                sys.exit(0)

    # Save the final model
    save_state(model, optimizer, scheduler, config.num_epochs - 1, tot_iter, config)
    # Write the status to the output directory to status.txt file
    with open(os.path.join(config.output, "status.txt"), "w") as f:
        f.write("finished")


if __name__ == "__main__":
    args = parse_args()
    # print command line args
    print(args)
    config = load_config(args.config)

    # Update config with command line arguments
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    # pretty print the config
    for key, value in config.items():
        print(f"{key}: {value}")

    # Set the random seed
    if config.seed is not None:
        set_seed(config.seed)

    if config.output is None:
        config.output = config.log_dir

    out_path = Path(config.output)
    out_path.mkdir(parents=True, exist_ok=True)

    train(config, device=args.device)
