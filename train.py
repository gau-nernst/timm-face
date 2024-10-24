import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import timm.optim
import torch
import wandb
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import KFold
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import InsightFaceBinDataset, create_train_dloader
from ema import EMA
from modelling import TimmFace

logger = logging.getLogger()
logger.setLevel(logging.INFO)
_LOGGING_FORMATTER = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s")
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(_LOGGING_FORMATTER)
logger.addHandler(_stdout_handler)


class CosineSchedule:
    def __init__(self, lr: float, total_steps: int, warmup: float = 0.05, decay_multiplier: float = 1e-2) -> None:
        self.lr = lr
        self.final_lr = lr * decay_multiplier
        self.total_steps = total_steps
        self.warmup_steps = round(total_steps * warmup)

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * step / self.warmup_steps
        if step < self.total_steps:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.final_lr + 0.5 * (self.lr - self.final_lr) * (1 + math.cos(progress * math.pi))
        return self.final_lr

    def set_lr(self, step: int, optim):
        lr = self.get_lr(step)
        for group in optim.param_groups:
            if isinstance(group["lr"], Tensor):
                group["lr"].copy_(lr)
            else:
                group["lr"] = lr
        return lr


# adapted from https://github.com/deepinsight/insightface/blob/v0.7/recognition/arcface_torch/eval/verification.py
def kfold_accuracy(y_true: np.ndarray, y_score: np.ndarray, n_folds: int = 10):
    kfold = KFold(n_folds)
    accs = []

    for train_indices, test_indices in kfold.split(np.arange(y_true.shape[0])):
        y_true_train = y_true[train_indices]
        y_score_train = y_score[train_indices]

        _, _, thresholds = roc_curve(y_true_train, y_score_train)
        pred_train = y_score_train >= thresholds[:, None]  # (n_thresholds, fold_size)
        acc_train = (pred_train == y_true_train).sum(1) / y_true_train.shape[0]
        optimal_th = thresholds[np.argmax(acc_train)]

        accs.append(accuracy_score(y_true[test_indices], y_score[test_indices] >= optimal_th))

    return np.mean(accs)


def build_optim(
    model: nn.Module, optim: str, lr: float, weight_decay: float, param_groups: list[dict] | None = None, **kwargs
):
    _globals = dict(torch=torch, timm=timm)
    try:
        import torchao.prototype.low_bit_optim

        _globals["torchao"] = torchao
    except ImportError:
        pass
    optim_cls = eval(optim, _globals)

    def _match_prefix(name: str, prefix: str):
        name_parts = name.split(".")
        prefix_parts = prefix.split(".")
        return name_parts[:prefix_parts] == prefix_parts

    if param_groups is not None:
        groups = []
        for group in param_groups:
            group = dict(group)  # shallow copy
            prefix = group.pop(prefix)
            group["params"] = [p for name, p in model.named_parameters() if _match_prefix(name, prefix)]
            logger.info(f"  - {prefix=}: {sum(p.numel() for p in group['params']):,} params")
            groups.append(group)

        other_params = [p for p in model.parameters() if all(p not in group["params"] for group in groups)]
        logger.info(f"  - others: {sum(p.numel() for p in other_params)}")
        groups.append(dict(params=other_params))

    else:
        groups = list(model.parameters())

    return optim_cls(groups, lr=lr, weight_decay=weight_decay, **kwargs)


def amp_ctx(amp_dtype: torch.dtype | None):
    return torch.autocast("cuda", amp_dtype, amp_dtype is not None)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--backbone_kwargs", type=json.loads, default=dict())
    parser.add_argument("--n_classes", type=int, default=93_431)  # MS1MV3
    parser.add_argument("--loss", default="cosface")
    parser.add_argument("--loss_kwargs", type=json.loads, default=dict())
    parser.add_argument("--reduce_first_conv_stride", action="store_true")
    parser.add_argument("--partial_fc", type=int, default=0)

    def _get_dtype(x: str):
        return dict(fp32=torch.float32, bf16=torch.bfloat16)[x]

    parser.add_argument("--model_dtype", type=_get_dtype, default=torch.float32)
    parser.add_argument("--amp_dtype", type=_get_dtype)
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=1000)

    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--augmentations", nargs="+")
    parser.add_argument("--val_ds", nargs="+", type=Path)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=4)

    parser.add_argument("--optim", default="torch.optim.AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--param_groups", type=json.loads)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())

    parser.add_argument("--clip_grad_norm", type=float)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--decay_multiplier", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=1)

    parser.add_argument("--run_name", default="debug")
    parser.add_argument("--resume")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.model_dtype != torch.float32:
        assert args.amp_dtype is None, "AMP should not be used when model is FP16/BF16"
    args.torch_version = torch.__version__

    # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    # https://pytorch.org/docs/stable/elastic/run.html
    is_ddp = os.environ.get("RANK") is not None
    if is_ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        is_master = int(os.environ["RANK"]) == 0
        torch.cuda.set_device(local_rank)

        world_size = int(os.environ["WORLD_SIZE"])
        assert args.batch_size % world_size == 0
        batch_size = args.batch_size // world_size

    else:
        is_master = True
        batch_size = args.batch_size

    if is_master:
        for k, v in vars(args).items():
            logger.info(f"{k}: {v}")

        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        CKPT_DIR = Path("checkpoints") / f"{args.run_name}_{time_now}"
        assert not CKPT_DIR.exists()
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        wandb.init(project="Timm Face", name=args.run_name, config=args, dir="/tmp")

    assert batch_size % args.grad_accum == 0
    dloader, train_size = create_train_dloader(
        args.ds_path,
        batch_size // args.grad_accum,
        augmentations=args.augmentations,
        n_workers=args.n_workers,
        device="cuda",
    )
    if is_master:
        logger.info(f"Train dataset: {train_size:,} images")
        logger.info(f"{args.total_steps / (train_size // args.batch_size):.2f} epochs")
        val_ds_paths = sorted(Path(args.ds_path).glob("*.bin")) if args.val_ds is None else args.val_ds

    model = TimmFace(
        args.backbone,
        args.n_classes,
        args.loss,
        backbone_kwargs=args.backbone_kwargs,
        loss_kwargs=args.loss_kwargs,
        reduce_first_conv_stride=args.reduce_first_conv_stride,
        partial_fc=args.partial_fc,
    )
    for p in model.parameters():
        p.data = p.detach().to(args.model_dtype)  # only cast params, don't cast buffers
    model.cuda()
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    if args.compile:
        model.compile()
    if is_master:
        ema = EMA(model)
        logger.info("Model parameters:")
        logger.info(f"  Backbone: {sum(p.numel() for p in model.backbone.parameters()):,}")
        logger.info(f"  Head: {model.weight.numel():,}")

    optim = build_optim(model, args.optim, args.lr, args.weight_decay, args.param_groups, **args.optim_kwargs)
    lr_schedule = CosineSchedule(args.lr, args.total_steps, warmup=args.warmup, decay_multiplier=args.decay_multiplier)
    step = 0

    if args.resume is not None and is_master:
        logger.info(f"Resume from {args.resume}")
        ckpt = torch.load(args.resume)
        step = ckpt["step"]
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optim.load_state_dict(ckpt["optim"])

    if is_ddp:
        # this will broadcast weights to other processes at init
        nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

        step_tensor = torch.tensor(step, device="cuda")
        dist.broadcast(step_tensor, 0)
        step = step_tensor.item()

    pbar = tqdm(total=args.total_steps, dynamic_ncols=True, initial=step, disable=not is_master)
    model.train()
    time0 = time.perf_counter()
    log_interval = 100

    while step < args.total_steps:
        for _ in range(args.grad_accum):
            images, labels = next(dloader)
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            with amp_ctx(args.amp_dtype):
                loss, norms = model(images, labels)
            (loss / args.grad_accum).backward()

        lr = lr_schedule.set_lr(step, optim)
        grad_norm = None
        if args.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

        if step % log_interval == 0:
            loss = loss.detach()
            if is_ddp:
                dist.all_reduce(loss, dist.ReduceOp.AVG)
            if is_master:
                if grad_norm is None:
                    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                    grad_norms = torch._foreach_norm(grads)
                    grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms, dim=0))
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
        optim.zero_grad()
        step += 1
        pbar.update()

        if is_master:
            if step % log_interval == 0:
                time1 = time.perf_counter()
                log_dict = dict(
                    max_memory_allocated=torch.cuda.max_memory_allocated(),
                    imgs_seen_millions=args.batch_size * step / 1e6,
                    imgs_per_second=args.batch_size * log_interval / (time1 - time0),
                )
                wandb.log(log_dict, step=step)
                time0 = time1

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
                        with torch.no_grad(), amp_ctx(args.amp_dtype):
                            embs1 = ema(imgs1.cuda()).float()
                            embs2 = ema(imgs2.cuda()).float()
                        all_scores.append((embs1 * embs2).sum(1).cpu().numpy())

                    all_labels = np.concatenate(all_labels, axis=0)
                    all_scores = np.concatenate(all_scores, axis=0)

                    acc = kfold_accuracy(all_labels, all_scores)
                    wandb.log({f"acc/{val_ds_name}": acc}, step=step)

                checkpoint = dict(
                    step=step,
                    model=model.state_dict(),
                    ema=ema.state_dict(),
                )
                torch.save(checkpoint, CKPT_DIR / f"step_{step}.pth")
                checkpoint.update(optim=optim.state_dict())
                torch.save(checkpoint, CKPT_DIR / "last.pth")  # for resume, w/ optim states
                model.train()

    if is_ddp:
        dist.destroy_process_group()
