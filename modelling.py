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
        reduce_first_conv_stride: bool = False,
    ) -> None:
        super().__init__()
        EMBED_DIM = 512
        self.backbone = timm.create_model(backbone, num_classes=EMBED_DIM, **(backbone_kwargs or dict()))

        if reduce_first_conv_stride:
            first_conv = self.backbone
            for name in self.backbone.pretrained_cfg["first_conv"].split("."):
                first_conv = getattr(first_conv, name)
            first_conv.stride = tuple(s // 2 for s in first_conv.stride)

        self.bn = nn.BatchNorm1d(EMBED_DIM, affine=False)  # this is important
        self.weight = nn.Parameter(torch.empty(n_classes, EMBED_DIM).normal_(0, 0.01))

        loss_lookup = dict(adaface=AdaFace, arcface=ArcFace, cosface=CosFace)
        self.loss = loss_lookup[loss](**(loss_kwargs or dict()))

    def forward(self, imgs: Tensor, labels: Tensor | None = None) -> Tensor:
        embs = self.bn(self.backbone(imgs))
        if not self.training:
            return F.normalize(embs, dim=1)

        weight, labels = partialfc_sample(self.weight, labels, 16_384)
        logits = F.normalize(embs, dim=1) @ F.normalize(weight, dim=1).T

        norms = torch.linalg.vector_norm(embs.detach(), dim=1)
        loss = self.loss(logits.float(), norms, labels)
        return loss, norms


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
        self.register_buffer("norm_mean", torch.tensor(20.0))
        self.register_buffer("norm_std", torch.tensor(100.0))

    def forward(self, logits: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        norms = norms.clip(0.001, 100)
        std, mean = torch.std_mean(norms)
        self.norm_mean.lerp_(mean, 1e-2)
        self.norm_std.lerp_(std, 1e-2)

        margin_scaler = (norms - self.norm_mean) / (self.norm_std + 1e-3)
        margin_scaler = (margin_scaler * self.h).clip(-1.0, 1.0)

        positives = logits[torch.arange(logits.shape[0]), labels]
        theta = positives.clamp(-0.999, 0.999).acos() - self.m * margin_scaler  # g_angle
        positives = theta.clip(0, torch.pi).cos() - self.m * (1.0 + margin_scaler)  # g_add
        logits[torch.arange(logits.shape[0]), labels] = positives
        return F.cross_entropy(logits * self.s, labels)


class ArcFace(nn.Module):
    def __init__(self, s: float = 64.0, m: float = 0.5) -> None:
        super().__init__()
        self.m = m
        self.s = s

    def forward(self, logits: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        positives = logits[torch.arange(logits.shape[0]), labels]
        theta = positives.clamp(-0.999, 0.999).acos() + self.m
        positives = theta.clip(0.0, torch.pi).cos()
        logits[torch.arange(logits.shape[0]), labels] = positives
        return F.cross_entropy(logits * self.s, labels)


class CosFace(nn.Module):
    def __init__(self, s: float = 64.0, m: float = 0.4) -> None:
        super().__init__()
        self.m = m
        self.s = s

    def forward(self, logits: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        logits[torch.arange(logits.shape[0]), labels] -= self.m
        return F.cross_entropy(logits * self.s, labels)
