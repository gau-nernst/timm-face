import argparse
import json
from pathlib import Path

import timm
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import InsightFaceDataset
from modelling import Head, build_loss


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_kwargs", type=json.loads, default=dict())
    parser.add_argument("--loss", default="adaface")
    parser.add_argument("--n_epochs", type=int, default=1)

    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--betas", type=json.loads, default=[0.9, 0.95])

    parser.add_argument("--run_name", default="debug")
    return parser


if __name__ == "__main__":
    DEVICE = "cuda"
    EMBED_DIM = 512

    args = get_parser().parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    Path("wandb_logs").mkdir(exist_ok=True)
    wandb.init(project="Timm Face", name=args.run_name, config=args, dir="wandb_logs")

    ds = InsightFaceDataset(args.ds_path)
    dloader = DataLoader(
        ds,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = timm.create_model(args.model, num_classes=EMBED_DIM, **args.model_kwargs).to(DEVICE)
    head = Head(EMBED_DIM, ds.n_classes).to(DEVICE)
    criterion = build_loss(args.loss).to(DEVICE)

    optim = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )

    step = 0
    for epoch_idx in range(args.n_epochs):
        for images, labels in tqdm(dloader, dynamic_ncols=True, desc=f"Epoch {epoch_idx + 1}/{args.n_epochs}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.autocast("cuda", torch.bfloat16):
                embs = model(images)
                logits, norms, labels = head(embs, labels)
                loss = criterion(logits, norms, labels)

            if step % 100 == 0:
                wandb.log(dict(loss=loss.item()), step=step)

            loss.backward()
            optim.step()
            optim.zero_grad()

            step += 1
