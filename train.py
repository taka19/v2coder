#!/usr/bin/env python3
import argparse
import os
import time
import yaml

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import utils
from dataset import get_file_list, AudioSegmentDataset, Collator
import hvae
from model import Model


class ModelWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def load_config(args):
    with open(args.config_file, "r") as f:
        cfg = yaml.safe_load(f)

    def apply_recursive(obj, func):
        if isinstance(obj, dict):
            return {k: apply_recursive(v, func) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [apply_recursive(v, func) for v in obj]
        elif isinstance(obj, str):
            return func(obj)
        return obj

    cfg = apply_recursive(cfg, os.path.expandvars)

    cfg["model"].setdefault("msg", {})
    cfg["model"]["msg"].setdefault("channels", cfg["data"]["msg"]["num_mel_bins"])
    cfg["model"]["msg"].setdefault(
        "parameters",
        {"sample_rate": cfg["data"]["sample_rate"]} | cfg["data"]["stft"] | cfg["data"]["msg"],
    )
    cfg["model"].setdefault("wav", {})
    cfg["model"]["wav"].setdefault("channels", 1)

    config_file = os.path.join(args.checkpoint_dir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f)
    return cfg

def train_step(rank, cfg, model, optimizers, epoch, step, batch, writer):
    start = time.perf_counter()

    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)

    wav, msg, = (x.cuda(rank, non_blocking=True) for x in batch)
    loss, outputs, reconstruction = model(
        wav, msg,
        constraint=cfg["constraint"],
        skip_weight=max(0.0, 1.0 - (step + 1) / cfg["skip_duration"]),
    )
    loss.backward()

    if rank == 0 and step % cfg["logging"]["summary"] == 0:
        with torch.no_grad():
            distortion, rate, rate_constraint, = outputs
            negative_elbo = distortion + rate
            log_precision = -torch.log(torch.square(wav - reconstruction).mean())

        scalar_dict = {
            "distortion": distortion,
            "rate": rate,
            "rate_constraint": rate_constraint,
            "elbo": -negative_elbo,
            "loss": loss,
            "log_precision": log_precision,
        }
        for key, value in scalar_dict.items():
            writer.add_scalar(f"training/{key}", value, step)

    if rank == 0 and step % cfg["logging"]["stdout"] == 0:
        distortion, rate, rate_constraint, = outputs
        print(
            "Epoch=%d Step=%d Loss=%.6e Lr=%.6e %.3f s"
            % (epoch, step, loss, optimizers["vae"].param_groups[0]["lr"], time.perf_counter() - start)
        )

    for optimizer in optimizers.values():
        optimizer.step()

def set_reproducibility(seed=12345):
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

def get_dataloader(cfg, num_gpus, rank):
    training_data_list = get_file_list(cfg["data"]["root"], cfg["data"]["training_file"])
    trainset = AudioSegmentDataset(
        training_data_list,
        segment_size=cfg["train"]["train_segment_size"],
    )
    collate_fn = Collator(
        sample_rate=cfg["data"]["sample_rate"],
        **cfg["data"]["stft"], **cfg["data"]["msg"],
    )
    train_sampler = None
    if 1 < num_gpus:
        train_sampler = DistributedSampler(trainset, num_replicas=num_gpus, rank=rank, shuffle=True, drop_last=True)

    num_workers_per_gpu = cfg["train"]["num_workers"] // num_gpus
    batch_size_per_gpu = cfg["train"]["batch_size"] // num_gpus
    train_dataloader = DataLoader(
        trainset, num_workers=num_workers_per_gpu,
        sampler=train_sampler, shuffle=train_sampler is None,
        batch_size=batch_size_per_gpu,
        collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    return train_dataloader, train_sampler

def train(rank, num_gpus, args, cfg):
    if 1 < num_gpus:
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=num_gpus, rank=rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_reproducibility()

    is_saved = False
    model, optimizers, schedulers, epoch, step = build(cfg, device)
    if args.checkpoint_file is not None:
        model, optimizers, schedulers, epoch, step = load_checkpoint(
            args.checkpoint_file, device, model, optimizers, schedulers, epoch, step,
        )
        is_saved = True

    writer = None
    if rank == 0:
        writer = SummaryWriter(os.path.join(args.checkpoint_dir, "logs"))
        #print(model)
        print(
            "Number of model parameters: %d (G: %d, I: %d) " %
            (
                utils.num_params(model.parameters()),
                utils.num_params(model.vae.generative_model.parameters()),
                utils.num_params(model.vae.inference_model.parameters()),
            )
        )

    train_dataloader, train_sampler = get_dataloader(cfg, num_gpus, rank)
    model = DDP(model, device_ids=[rank]).to(device) if 1 < num_gpus else ModelWrapper(model).to(device)
    model.train()

    while epoch < cfg["train"]["max_epochs"]:
        if train_sampler:
            train_sampler.set_epoch(epoch)
        for batch in train_dataloader:
            if not is_saved and rank == 0 and (step == 10000 or step % 50000 == 0):
                save_checkpoint(args.checkpoint_dir, "latest", model, optimizers, schedulers, epoch, step)

            if not is_saved and rank == 0 and step % cfg["train"]["logging"]["checkpoint"] == 0:
                save_checkpoint(args.checkpoint_dir, str(step), model, optimizers, schedulers, epoch, step)
                is_saved = True

            train_step(rank, cfg["train"], model, optimizers, epoch, step, batch, writer)
            step += 1
            is_saved = False
        else:
            for scheduler in schedulers.values():
                scheduler.step()
            epoch += 1

    if not is_saved and rank == 0:
        save_checkpoint(args.checkpoint_dir, "final", model, optimizers, schedulers, epoch, step)
    if 1 < num_gpus:
        torch.distributed.destroy_process_group()

def save_checkpoint(ckpt_dir, ckpt_name, model, optimizers, schedulers, epoch, step):
    ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}.ckpt")
    print(f"Saving checkpoint to {ckpt_path}")
    torch.save({
        "model": model.module.state_dict(),
        "optimizers": {key: optimizer.state_dict() for key, optimizer in optimizers.items()},
        "schedulers": {key: scheduler.state_dict() for key, scheduler in schedulers.items()},
        "epoch": epoch,
        "step": step,
    }, ckpt_path)

def load_checkpoint(checkpoint_path, device, model, optimizers, schedulers, epoch, step):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    for key in optimizers.keys():
        optimizers[key].load_state_dict(checkpoint["optimizers"][key])
    for key in schedulers.keys():
        schedulers[key].load_state_dict(checkpoint["schedulers"][key])
    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    return model, optimizers, schedulers, epoch, step

def build(cfg, device):
    model = Model(cfg["model"]).to(device)
    optimizers = {
        key: AdamW(model[key].parameters(), **params)
        for key, params in cfg["train"]["optimizer"].items()
    }
    schedulers = {
        key: torch.optim.lr_scheduler.StepLR(optimizers[key], **params)
        for key, params in cfg["train"]["scheduler"].items()
    }
    epoch = 0
    step = 0
    return model, optimizers, schedulers, epoch, step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--checkpoint_file", type=str, default=None,
                        help="file path to checkpoint to restore")
    args = parser.parse_args()
    assert torch.cuda.is_available()

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print("Checkpoint directory: %s" % args.checkpoint_dir)

    cfg = load_config(args)
    num_gpus = torch.cuda.device_count()
    if 1 < num_gpus:
        mp.spawn(train, nprocs=num_gpus, args=(num_gpus, args, cfg,), join=True)
    else:
        train(0, num_gpus, args, cfg)

if __name__ == "__main__":
    main()
