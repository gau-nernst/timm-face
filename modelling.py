# references:
# https://github.com/mk-minchul/AdaFace/blob/master/head.py
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def build_head(name: str, embed_dim: int, n_classes: int, **kwargs):
    return dict(adaface=AdaFace, arcface=ArcFace, cosface=CosFace)[name](embed_dim, n_classes, **kwargs)


class AdaFace(nn.Linear):
    def __init__(self, embed_dim: int, n_classes: int, m: float = 0.4, h: float = 0.333, s: float = 64.0) -> None:
        super().__init__(embed_dim, n_classes, bias=False)
        nn.init.normal_(self.weight, 0, 0.01)
        self.m = m
        self.h = h
        self.s = s
        self.norm_normalizer = nn.BatchNorm1d(1, eps=1e-3, momentum=0.01, affine=False)
        nn.init.constant_(self.norm_normalizer.running_mean, 20.0)
        nn.init.constant_(self.norm_normalizer.running_var, 100.0 * 100.0)

    def forward(self, embs: Tensor, labels: Tensor) -> Tensor:
        norms = torch.linalg.vector_norm(embs, dim=1, keepdim=True)
        logits = (embs / norms) @ F.normalize(self.weight, dim=1).T

        with torch.autocast("cuda", enabled=False):
            margin_scaler = self.norm_normalizer(norms).squeeze(1)
            margin_scaler = (margin_scaler * self.h).clip(-1.0, 1.0)

            theta = logits.float().clamp(-0.999, 0.999).acos()
            theta[torch.arange(logits.shape[0]), labels] -= self.m * margin_scaler  # g_angle
            logits = theta.clip(0.0, torch.pi).cos()

            logits[torch.arange(logits.shape[0]), labels] -= (1.0 + self.m) * margin_scaler  # g_add
            return F.cross_entropy(logits * self.s, labels)


class ArcFace(nn.Linear):
    def __init__(self, embed_dim: int, n_classes: int, s: float = 64.0, m: float = 0.5) -> None:
        super().__init__(embed_dim, n_classes, bias=False)
        nn.init.normal_(self.weight, 0, 0.01)
        self.m = m
        self.s = s

    def forward(self, embs: Tensor, labels: Tensor) -> Tensor:
        logits = F.normalize(embs, dim=1) @ F.normalize(self.weight, dim=1).T

        with torch.autocast("cuda", enabled=False):
            theta = logits.float().clamp(-0.999, 0.999).acos()
            theta[torch.arange(logits.shape[0]), labels] += self.m
            logits = theta.cos()
            return F.cross_entropy(logits * self.s, labels)


class CosFace(nn.Linear):
    def __init__(self, embed_dim: int, n_classes: int, s: float = 64.0, m: float = 0.4) -> None:
        super().__init__(embed_dim, n_classes, bias=False)
        nn.init.normal_(self.weight, 0, 0.01)
        self.m = m
        self.s = s

    def forward(self, embs: Tensor, labels: Tensor) -> Tensor:
        logits = F.normalize(embs, dim=1) @ F.normalize(self.weight, dim=1).T

        with torch.autocast("cuda", enabled=False):
            logits[torch.arange(logits.shape[0]), labels] -= self.m
            return F.cross_entropy(logits * self.s, labels)
