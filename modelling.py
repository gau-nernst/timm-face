# references:
# https://github.com/mk-minchul/AdaFace/blob/master/head.py
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TimmFace(nn.Module):
    def __init__(
        self,
        backbone: str,
        n_classes: int,
        loss: str,
        backbone_kwargs: dict | None = None,
        loss_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        EMBED_DIM = 512
        self.backbone = timm.create_model(backbone, num_classes=EMBED_DIM, **(backbone_kwargs or dict()))
        self.bn = nn.BatchNorm1d(EMBED_DIM, affine=False)  # this is important
        self.weight = nn.Parameter(torch.randn(n_classes, EMBED_DIM))
        self.weight.data = F.normalize(self.weight.data, dim=1)

        loss_lookup = dict(adaface=AdaFace, arcface=ArcFace, cosface=CosFace)
        self.loss = loss_lookup[loss](**(loss_kwargs or dict()))

    def forward(self, imgs: Tensor, labels: Tensor | None = None) -> Tensor:
        embs = self.bn(self.backbone(imgs))
        if not self.training:
            return F.normalize(embs, dim=1)

        norms = torch.linalg.vector_norm(embs, dim=1, keepdim=True)
        weight, labels = partialfc_sample(self.weight, labels, 16_384)
        logits = (embs / norms) @ F.normalize(weight, dim=1).T

        return self.loss(logits.float(), norms, labels), norms


def partialfc_sample(weight: Tensor, labels: Tensor, n: int) -> tuple[Tensor, Tensor]:
    positives = torch.unique(labels, sorted=True)
    if n >= positives.shape[0]:
        perm = torch.rand(weight.shape[0], device=weight.device)
        perm[positives] = 2.0
        indices = torch.topk(perm, k=n)[1]
        indices = indices.sort()[0]
    else:
        indices = positives

    labels = torch.searchsorted(indices, labels)
    return weight[indices], labels


class AdaFace(nn.Module):
    def __init__(self, m: float = 0.4, h: float = 0.333, s: float = 64.0) -> None:
        super().__init__()
        self.m = m
        self.h = h
        self.s = s
        self.norm_normalizer = nn.BatchNorm1d(1, eps=1e-3, momentum=0.01, affine=False)
        nn.init.constant_(self.norm_normalizer.running_mean, 20.0)
        nn.init.constant_(self.norm_normalizer.running_var, 100.0 * 100.0)

    def forward(self, logits: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        margin_scaler = self.norm_normalizer(norms).squeeze(1)
        margin_scaler = (margin_scaler * self.h).clip(-1.0, 1.0)

        theta = logits.clamp(-0.999, 0.999).acos()
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
        theta = logits.clamp(-0.999, 0.999).acos()
        theta[torch.arange(logits.shape[0]), labels] += self.m
        logits = theta.cos()
        return F.cross_entropy(logits * self.s, labels)


class CosFace(nn.Module):
    def __init__(self, s: float = 64.0, m: float = 0.4) -> None:
        super().__init__()
        self.m = m
        self.s = s

    def forward(self, logits: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        logits[torch.arange(logits.shape[0]), labels] -= self.m
        return F.cross_entropy(logits * self.s, labels)
