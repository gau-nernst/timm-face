import copy
import itertools

import torch
from torch import nn


def params_buffers(m: nn.Module):
    return itertools.chain(m.parameters(), m.buffers())


class EMA(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        warmup_steps: int = 100,
        update_interval: int = 10,
        beta: float = 0.999,
    ) -> None:
        super().__init__()
        self.ema_model = copy.deepcopy(model)
        self.model = [model]  # not included in state dict
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.beta = beta

    @torch.no_grad()
    def update(self, step: int) -> None:
        if step < self.warmup_steps or (step - self.warmup_steps) % self.update_interval:
            return

        if step == self.warmup_steps:
            for ema_p, p in zip(params_buffers(self.ema_model), params_buffers(self.model[0])):
                ema_p.copy_(p)
            return

        for ema_p, p in zip(params_buffers(self.ema_model), params_buffers(self.model[0])):
            if ema_p.is_floating_point():
                ema_p.lerp_(p, 1.0 - self.beta)
            else:
                ema_p.copy_(p)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.ema_model.load_state_dict(state_dict, strict, assign)
