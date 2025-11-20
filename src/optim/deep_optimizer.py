import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Iterable, List, Dict, Tuple, Callable, Optional


class _CoordWiseMemory(nn.Module):
    def __init__(self, in_features: int = 6, hidden: int = 32, dtype=None, device=None):
        super().__init__()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=True)
        self.fc1  = nn.Linear(in_features, hidden)
        self.act  = nn.ReLU()
        self.fc2  = nn.Linear(hidden, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.norm(feat)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.squeeze(-1)

class DeepOptimizer(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        *,
        meta_lr: float = 1e-3,
        alpha: float = 0.99,
        beta1: float = 0.9, 
        beta2: float = 0.999,
        eps: float = 1e-8,
        hidden: int = 32,
        chunk_size: int = 262_144,     
        meta_objective: str = "alignment",
        grad_clip_meta: Optional[float] = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid lr")
        if meta_lr <= 0.0:
            raise ValueError("Invalid meta_lr")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        ref_param = None
        for group in self.param_groups:
            if len(group["params"]) > 0:
                ref_param = group["params"][0]
                break
        if ref_param is None:
            raise ValueError("No parameters passed to optimizer.")
        self._device = device or ref_param.device
        self._dtype  = dtype or ref_param.dtype

        self.meta_lr = float(meta_lr)
        self.alpha = float(alpha)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.chunk_size = int(chunk_size)
        self.meta_objective = meta_objective
        self.grad_clip_meta = grad_clip_meta
        self.memory = _CoordWiseMemory(in_features=6, hidden=hidden,
                                       device=self._device, dtype=self._dtype)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["ema1"] = torch.zeros_like(p.data, device=self._device, dtype=self._dtype)
                state["ema2"] = torch.zeros_like(p.data, device=self._device, dtype=self._dtype)

        self._step = 0

    def _build_features(
        self, g_flat: torch.Tensor, ema1_flat: torch.Tensor, ema2_flat: torch.Tensor
    ) -> torch.Tensor:
        eps = self.eps
        abs_g = g_flat.abs()
        feat = torch.stack(
            [
                g_flat,
                abs_g,
                (abs_g + eps).log(),
                ema1_flat,
                (ema2_flat.sqrt() + eps),
                torch.sign(g_flat),
            ],
            dim=-1,
        )
        return feat

    def _meta_loss_chunk(self, out_chunk: torch.Tensor, g_chunk: torch.Tensor) -> torch.Tensor:
        if self.meta_objective == "alignment":
            return (out_chunk * g_chunk).sum()
        elif self.meta_objective == "mse":
            return ((out_chunk + g_chunk) ** 2).mean()
        else:
            raise ValueError(f"Unknown meta_objective: {self.meta_objective}")

    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _outer_update(self, group_lr: float, p: torch.nn.Parameter, upd_flat: torch.Tensor):
        upd = upd_flat.view_as(p.data)
        p.data.add_(upd, alpha=group_lr)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                self.zero_grad(set_to_none=True)
                loss = closure()
                if not torch.is_tensor(loss):
                    raise RuntimeError("closure must return a torch.Tensor loss")
                loss.backward()
        for p_m in self.memory.parameters():
            if p_m.grad is not None:
                p_m.grad.zero_()

        meta_loss_total = torch.zeros((), device=self._device, dtype=self._dtype)
        for group in self.param_groups:
            group_lr = float(group["lr"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                st = self.state[p]
                ema1 = st["ema1"]
                ema2 = st["ema2"]
                ema1.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                ema2.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                g_flat    = g.reshape(-1).to(self._device, self._dtype)
                ema1_flat = ema1.reshape(-1).to(self._device, self._dtype)
                ema2_flat = ema2.reshape(-1).to(self._device, self._dtype)

                N = g_flat.numel()
                if N == 0:
                    continue
                start = 0
                while start < N:
                    end = min(start + self.chunk_size, N)
                    g_chunk    = g_flat[start:end]
                    ema1_chunk = ema1_flat[start:end]
                    ema2_chunk = ema2_flat[start:end]

                    feat = self._build_features(g_chunk, ema1_chunk, ema2_chunk)  # [M, 6]
                    out  = self.memory(feat)                                      # [M]

                    meta_loss_total = meta_loss_total + self._meta_loss_chunk(out, g_chunk)
                    start = end

        meta_loss_total.backward()

        if self.grad_clip_meta is not None:
            torch.nn.utils.clip_grad_norm_(self.memory.parameters(), self.grad_clip_meta)
        with torch.no_grad():
            for p_m in self.memory.parameters():
                g_m = p_m.grad if p_m.grad is not None else 0.0
                p_m.mul_(self.alpha).add_(g_m, alpha=-self.meta_lr)

        with torch.no_grad():
            for group in self.param_groups:
                group_lr = float(group["lr"])
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    g = p.grad.detach()
                    st = self.state[p]
                    ema1 = st["ema1"]
                    ema2 = st["ema2"]

                    g_flat    = g.reshape(-1).to(self._device, self._dtype)
                    ema1_flat = ema1.reshape(-1).to(self._device, self._dtype)
                    ema2_flat = ema2.reshape(-1).to(self._device, self._dtype)

                    N = g_flat.numel()
                    if N == 0:
                        continue

                    upd_flat = torch.empty_like(g_flat)

                    start = 0
                    while start < N:
                        end = min(start + self.chunk_size, N)
                        feat = self._build_features(
                            g_flat[start:end], ema1_flat[start:end], ema2_flat[start:end]
                        )
                        upd_flat[start:end] = self.memory(feat)  # không cần grad ở outer
                        start = end

                    self._outer_update(group_lr, p, upd_flat)

        self._step += 1
        return loss

    def state_dict(self):
        base = super().state_dict()
        base["dmgd_memory_state"] = self.memory.state_dict()
        base["dmgd_hparams"] = dict(
            meta_lr=self.meta_lr,
            alpha=self.alpha,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            chunk_size=self.chunk_size,
            meta_objective=self.meta_objective,
            grad_clip_meta=self.grad_clip_meta,
            step=self._step,
        )
        return base

    def load_state_dict(self, state_dict):
        self.memory.load_state_dict(state_dict.pop("dmgd_memory_state"))
        dm = state_dict.pop("dmgd_hparams", {})
        for k, v in dm.items():
            setattr(self, k, v)
        super().load_state_dict(state_dict)