import argparse
import json
import math
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import InsightFaceBinDataset, InsightFaceRecordIoDataset
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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--backbone_kwargs", type=json.loads, default=dict())
    parser.add_argument("--loss", default="adaface")
    parser.add_argument("--loss_kwargs", type=json.loads, default=dict())

    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=1000)

    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--betas", type=json.loads, default=[0.9, 0.95])

    parser.add_argument("--run_name", default="debug")
    return parser


def cycle(dloader: DataLoader):
    while True:
        for batch in dloader:
            yield tuple(x.to("cuda") for x in batch)


if __name__ == "__main__":
    DEVICE = "cuda"

    args = get_parser().parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    Path("wandb_logs").mkdir(exist_ok=True)
    wandb.init(project="Timm Face", name=args.run_name, config=args, dir="wandb_logs")

    train_ds = InsightFaceRecordIoDataset(args.ds_path)
    dloader = DataLoader(
        train_ds,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dloader = cycle(dloader)

    val_ds_dict = {bin_path.stem: InsightFaceBinDataset(bin_path) for bin_path in Path(args.ds_path).glob("*.bin")}

    model = TimmFace(
        args.backbone,
        train_ds.n_classes,
        args.loss,
        backbone_kwargs=args.backbone_kwargs,
        loss_kwargs=args.loss_kwargs,
    ).to(DEVICE)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )
    lr_schedule = CosineSchedule(args.lr, args.total_steps)

    model.train()
    step = 0
    pbar = tqdm(total=args.total_steps, dynamic_ncols=True)

    for images, labels in dloader:
        lr = lr_schedule.get_lr(step)
        for param_group in optim.param_groups:
            param_group["lr"] = lr

        # TODO: grad accum
        with torch.autocast("cuda", torch.bfloat16):
            loss = model(images, labels)
        loss.backward()

        if step % 100 == 0:
            log_dict = dict(
                loss=loss.item(),
                lr=lr,
            )
            wandb.log(log_dict, step=step)

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()

        # TODO: eval and checkpoint

        if step == args.total_steps:
            break
