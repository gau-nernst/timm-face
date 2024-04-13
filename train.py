import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import InsightFaceBinDataset, create_train_dloader
from ema import EMA
from modelling import TimmFace


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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--backbone_kwargs", type=json.loads, default=dict())
    parser.add_argument("--loss", default="adaface")
    parser.add_argument("--loss_kwargs", type=json.loads, default=dict())
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=1000)

    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--betas", type=json.loads, default=[0.9, 0.95])
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--decay_multiplier", type=float, default=0.01)

    parser.add_argument("--run_name", default="debug")
    parser.add_argument("--resume")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    CKPT_DIR = Path("checkpoints") / f"{args.run_name}_{time_now}"
    assert not CKPT_DIR.exists()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    dloader, train_size, n_classes = create_train_dloader(args.ds_path, args.batch_size, args.n_workers, device="cuda")
    print(f"Train dataset: {train_size:,} images, {n_classes:,} ids")
    print(f"{args.total_steps / (train_size // args.batch_size):.2f} epochs")

    val_ds_paths = list(Path(args.ds_path).glob("*.bin"))
    val_ds_paths.sort()

    model = TimmFace(
        args.backbone,
        n_classes,
        args.loss,
        backbone_kwargs=args.backbone_kwargs,
        loss_kwargs=args.loss_kwargs,
    ).to("cuda")
    ema = EMA(model)
    print("Model parameters:")
    print(f"  Backbone: {sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"  Head: {model.weight.numel():,}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )
    lr_schedule = CosineSchedule(args.lr, args.total_steps, warmup=args.warmup, decay_multiplier=args.decay_multiplier)

    Path("wandb_logs").mkdir(exist_ok=True)
    wandb.init(project="Timm Face", name=args.run_name, config=args, dir="wandb_logs")

    model.train()
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    if args.compile:
        model.backbone.compile(fullgraph=True)

    step = 0

    if args.resume is not None:
        print(f"Resume from {args.resume}")
        ckpt = torch.load(args.resume)
        step = ckpt["step"]
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optim.load_state_dict(ckpt["optim"])

    pbar = tqdm(total=args.total_steps, dynamic_ncols=True, initial=step)

    for images, labels in dloader:
        lr = lr_schedule.get_lr(step)
        for param_group in optim.param_groups:
            param_group["lr"] = lr

        # TODO: grad accum
        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)
        with torch.autocast("cuda", torch.bfloat16):
            loss, norms = model(images, labels)
        loss.backward()

        if step % 100 == 0:
            norms = norms.detach().cpu().numpy()
            log_dict = dict(
                loss=loss.item(),
                lr=lr,
                norm_hist=wandb.Histogram(norms),
                norm_mean=norms.mean(),
            )
            wandb.log(log_dict, step=step)

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()
        ema.update(step)

        if step % args.eval_interval == 0:
            ema.eval()
            model.eval()

            for val_ds_path in val_ds_paths:
                val_ds_name = val_ds_path.stem
                val_ds = InsightFaceBinDataset(val_ds_path)
                val_dloader = DataLoader(val_ds, args.batch_size, num_workers=args.n_workers)

                all_labels = []
                all_scores = []

                for imgs1, imgs2, labels in tqdm(val_dloader, dynamic_ncols=True, desc=f"Evaluating {val_ds_name}"):
                    all_labels.append(labels.clone().numpy())
                    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                        embs1 = ema(imgs1.to("cuda"))
                        embs2 = ema(imgs2.to("cuda"))
                    all_scores.append((embs1 * embs2).sum(1).float().cpu().numpy())

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

        if step == args.total_steps:
            break
