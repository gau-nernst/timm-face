# references:
# https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_amp.py

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import timm.optim
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import InsightFaceBinDataset, create_train_dloader
from ema import EMA
from modelling import TimmFace
from train import CosineSchedule, get_parser, kfold_accuracy


if __name__ == "__main__":
    args = get_parser().parse_args()
    assert args.channels_last is False, "You cannot set --channels_last for XLA"
    assert args.compile is False
    assert args.amp_dtype in ("none", "bfloat16")

    device = xm.xla_device()
    is_rank0 = xm.get_ordinal() == 0
    batch_size = args.batch_size

    if is_rank0:
        for k, v in vars(args).items():
            print(f"{k}: {v}")

        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        CKPT_DIR = Path("checkpoints") / f"{args.run_name}_{time_now}"
        assert not CKPT_DIR.exists()
        CKPT_DIR.mkdir(parents=True, exist_ok=True)

        Path("wandb_logs").mkdir(exist_ok=True)
        wandb.init(project="Timm Face", name=args.run_name, config=args, dir="wandb_logs")

    assert batch_size % args.grad_accum == 0
    dloader, train_size = create_train_dloader(
        args.ds_path,
        batch_size // args.grad_accum,
        augmentations=args.augmentations,
        n_workers=args.n_workers,
        device=device,
    )
    if is_rank0:
        print(f"Train dataset: {train_size:,} images")
        print(f"{args.total_steps / (train_size // args.batch_size):.2f} epochs")
        val_ds_paths = sorted(Path(args.ds_path).glob("*.bin")) if args.val_ds is None else args.val_ds

    model = TimmFace(
        args.backbone,
        args.n_classes,
        args.loss,
        backbone_kwargs=args.backbone_kwargs,
        loss_kwargs=args.loss_kwargs,
        reduce_first_conv_stride=args.reduce_first_conv_stride,
        partial_fc=args.partial_fc,
    ).to(device)

    if is_rank0:
        ema = EMA(model)
        print("Model parameters:")
        print(f"  Backbone: {sum(p.numel() for p in model.backbone.parameters()):,}")
        print(f"  Head: {model.weight.numel():,}")

        if args.optim == "LAMB" and args.clip_grad_norm is not None:
            print("LAMB already has clip_grad_norm. Make sure this is intended.")

    optim_dict = dict(
        SGD=torch.optim.SGD,
        AdamW=torch.optim.AdamW,
        LAMB=timm.optim.Lamb,
    )
    optim = optim_dict[args.optim](
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        **(args.optim_kwargs or dict()),
    )
    lr_schedule = CosineSchedule(args.lr, args.total_steps, warmup=args.warmup, decay_multiplier=args.decay_multiplier)

    amp_dtype = dict(bfloat16=torch.bfloat16, none=None)[args.amp_dtype]
    amp_enabled = amp_dtype is not None

    step = 0

    if args.resume is not None:
        if is_rank0:
            print(f"Resume from {args.resume}")
        ckpt = torch.load(args.resume)
        step = ckpt["step"]
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optim.load_state_dict(ckpt["optim"])

    pbar = tqdm(total=args.total_steps, dynamic_ncols=True, initial=step, disable=not is_rank0)
    model.train()

    while step < args.total_steps:
        lr = lr_schedule.get_lr(step)
        for param_group in optim.param_groups:
            param_group["lr"] = lr

        for _ in range(args.grad_accum):
            images, labels = next(dloader)
            with torch.autocast("xla", amp_dtype, amp_enabled):
                loss, norms = model(images, labels)
            (loss / args.grad_accum).backward()

        if args.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        else:
            grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            grad_norms = torch._foreach_norm(grads)
            grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms, dim=0))

        if is_rank0 and step % 100 == 0:
            norms = norms.detach().cpu().numpy()
            log_dict = dict(
                loss=loss.item(),
                lr=lr,
                norm_hist=wandb.Histogram(norms),
                norm_mean=norms.mean(),
                grad_norm=grad_norm.item(),
            )
            wandb.log(log_dict, step=step)

        optim.step()
        xm.optimizer_step(optim)
        optim.zero_grad()

        step += 1
        pbar.update()

        if is_rank0:
            ema.update(step)

            if step % args.eval_interval == 0:
                ema.eval()
                model.eval()

                for val_ds_path in val_ds_paths:
                    val_ds_name = val_ds_path.stem
                    val_ds = InsightFaceBinDataset(str(val_ds_path))
                    val_dloader = DataLoader(val_ds, args.batch_size, num_workers=args.n_workers)

                    all_labels = []
                    all_scores = []

                    for imgs1, imgs2, labels in tqdm(val_dloader, dynamic_ncols=True, desc=f"Evaluating {val_ds_name}"):
                        all_labels.append(labels.clone().numpy())
                        with torch.no_grad(), torch.autocast("xla", amp_dtype, amp_enabled):
                            embs1 = ema(imgs1.to(device)).float()
                            embs2 = ema(imgs2.to(device)).float()
                        all_scores.append((embs1 * embs2).sum(1).cpu().numpy())

                    all_labels = np.concatenate(all_labels, axis=0)
                    all_scores = np.concatenate(all_scores, axis=0)

                    acc = kfold_accuracy(all_labels, all_scores)
                    wandb.log({f"acc/{val_ds_name}": acc}, step=step)

                checkpoint = {
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optim": optim.state_dict(),
                }
                torch.save(checkpoint, CKPT_DIR / f"step_{step}.pth")

                model.train()
