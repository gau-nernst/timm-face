# references:
# https://github.com/mk-minchul/AdaFace/blob/master/head.py
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Head(nn.Module):
    def __init__(self, embed_dim: int = 512, n_classes: int = 70722) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(n_classes, embed_dim))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, embs: Tensor, labels: Tensor) -> Tensor:
        norms = torch.linalg.vector_norm(embs, dim=-1)
        embs = embs / norms.unsqueeze(-1)
        logits = embs @ F.normalize(self.kernel, dim=-1).T
        return logits, norms, labels


def build_loss(name: str, **kwargs):
    return dict(adaface=AdaFace, arcface=ArcFace, cosface=CosFace)[name](**kwargs)


class AdaFace(nn.Module):
    eps = 1e-3

    def __init__(self, m: float = 0.4, h: float = 0.333, s: float = 64.0, t_alpha: float = 0.01) -> None:
        super().__init__()
        self.m = m
        self.h = h
        self.s = s
        self.norm_normalizer = nn.BatchNorm1d(1, eps=self.eps, momentum=t_alpha, affine=False)
        nn.init.constant_(self.norm_normalizer.running_mean, 20.0)
        nn.init.constant_(self.norm_normalizer.running_var, 100.0 * 100.0)

    def forward(self, logits: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        margin_scaler = self.norm_normalizer(norms.unsqueeze(-1)).squeeze(-1)
        margin_scaler = (margin_scaler * self.h).clip(-1.0, 1.0)

        theta = logits.acos()
        theta[torch.arange(logits.shape[0]), labels] -= self.m * margin_scaler  # g_angle
        logits = theta.clip(0.0, torch.pi).cos()

        logits[torch.arange(logits.shape[0]), labels] -= (1.0 + self.m) * margin_scaler  # g_add
        return F.cross_entropy(logits * self.s, labels)


class ArcFace(nn.Module):
    def __init__(self, s: float = 64.0, m: float = 0.5) -> None:
        super().__init__()
        self.m = m
        self.s = s

    def forward(self, logits: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        theta = logits.acos()
        theta[torch.arange(logits.shape[0]), labels] += self.m
        logits = theta.cos()
        return F.cross_entropy(logits * self.s, labels)


class CosFace(nn.Module):
    def __init__(self, s: float = 64.0, m: float = 0.4) -> None:
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, logits: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        logits[torch.arange(logits.shape[0]), labels] -= self.m
        return F.cross_entropy(logits * self.s, labels)
