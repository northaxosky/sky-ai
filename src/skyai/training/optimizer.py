"""Optimizer construction for AdamW and modern Muon split recipes"""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
import torch.nn as nn


class OptimizerChain:
    """Small wrapper that makes multiple optimizers look like one."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("OptimizerChain needs at least one optimizer")
        self.optimizers = optimizers
        self.param_groups = [group for opt in optimizers for group in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for opt, state in zip(self.optimizers, state_dict["optimizers"], strict=True):
            opt.load_state_dict(state)


class Muon(torch.optim.Optimizer):
    """Single-process Muon optimizer for 2D matrix parameters."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        name: str = "muon",
        momentum: float = 0.95,
        beta2: float = 0.9,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr,
            initial_lr=lr,
            momentum=momentum,
            beta2=beta2,
            weight_decay=weight_decay,
            initial_weight_decay=weight_decay,
            ns_steps=ns_steps,
            optimizer_type="muon",
            name=name,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon expects only 2D matrix parameters")

                grad = p.grad
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    rows, cols = p.shape
                    state["second_buffer"] = torch.zeros(
                        (rows, 1) if rows >= cols else (1, cols),
                        device=p.device,
                        dtype=p.dtype,
                    )

                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.lerp_(grad, 1.0 - momentum)
                update = grad.lerp(momentum_buffer, momentum)

                update = _polar_express(update, ns_steps=ns_steps)
                update = _normuon_scale(update, state["second_buffer"], beta2=beta2)

                rows, cols = p.shape
                scaled_lr = lr * math.sqrt(max(1.0, rows / cols))
                decay_mask = (update * p) >= 0
                p.sub_(scaled_lr * update + scaled_lr * weight_decay * p * decay_mask)
        return loss


def build_optimizer(
    model: nn.Module,
    *,
    optimizer_type: str = "adamw",
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    device_type: str = "cuda",
    total_batch_size: int = 524288,
    embedding_lr: float = 0.3,
    unembedding_lr: float = 0.008,
    matrix_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_beta2: float = 0.9,
    muon_ns_steps: int = 5,
) -> torch.optim.Optimizer | OptimizerChain:
    """Build AdamW with GPT-2's parameter group weight decay policy"""
    if optimizer_type == "muon-split":
        return build_muon_split_optimizer(
            model,
            embedding_lr=embedding_lr,
            unembedding_lr=unembedding_lr,
            matrix_lr=matrix_lr,
            muon_momentum=muon_momentum,
            muon_beta2=muon_beta2,
            muon_ns_steps=muon_ns_steps,
            weight_decay=weight_decay,
            total_batch_size=total_batch_size,
            device_type=device_type,
        )
    if optimizer_type != "adamw":
        raise ValueError(f"Unknown optimizer_type {optimizer_type!r}")

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Model has no trainable parameters")

    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay, "name": "adamw_decay"},
        {"params": nodecay_params, "weight_decay": 0.0, "name": "adamw_nodecay"},
    ]

    # Only pass fused= when AdamW supports it AND we're on cuda. Older PyTorch
    # builds without fused AdamW would TypeError if we always passed the kwarg.
    kwargs: dict[str, Any] = {"lr": learning_rate, "betas": betas, "eps": eps}
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    if fused_available and device_type == "cuda":
        kwargs["fused"] = True

    return torch.optim.AdamW(optim_groups, **kwargs)


def build_muon_split_optimizer(
    model: nn.Module,
    *,
    embedding_lr: float,
    unembedding_lr: float,
    matrix_lr: float,
    muon_momentum: float,
    muon_beta2: float,
    muon_ns_steps: int,
    weight_decay: float,
    total_batch_size: int,
    device_type: str = "cuda",
) -> OptimizerChain:
    batch_scale = (total_batch_size / 524288) ** 0.5
    dim_scale = (model.config.n_embed / 768) ** -0.5  # pyright: ignore[reportAttributeAccessIssue]

    embedding_params = [model.transformer.wte.weight]  # pyright: ignore[reportAttributeAccessIssue]
    lm_head_params = [] if model.config.tie_weights else [model.lm_head.weight]  # pyright: ignore[reportAttributeAccessIssue]
    reserved_ids = {id(p) for p in embedding_params + lm_head_params}

    muon_params: list[nn.Parameter] = []
    adamw_other: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in reserved_ids:
            continue
        if name.startswith("transformer.h.") and p.ndim == 2:
            muon_params.append(p)
        else:
            adamw_other.append(p)

    adam_groups = [
        _adam_group(
            "embed", embedding_params, embedding_lr * batch_scale * dim_scale, weight_decay=0.001
        ),
    ]
    if lm_head_params:
        adam_groups.append(
            _adam_group(
                "lm_head",
                lm_head_params,
                unembedding_lr * batch_scale * dim_scale,
                weight_decay=0.01,
            )
        )
    if adamw_other:
        adam_groups.append(
            _adam_group(
                "adamw_other",
                adamw_other,
                embedding_lr * batch_scale * dim_scale,
                weight_decay=0.0,
            )
        )

    kwargs: dict[str, Any] = {"betas": (0.8, 0.995), "eps": 1e-10}
    if "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda":
        kwargs["fused"] = True

    optimizers: list[torch.optim.Optimizer] = [torch.optim.AdamW(adam_groups, **kwargs)]

    by_shape: dict[tuple[int, ...], list[nn.Parameter]] = {}
    for p in muon_params:
        by_shape.setdefault(tuple(p.shape), []).append(p)
    for shape, params in by_shape.items():
        optimizers.append(
            Muon(
                params,
                lr=matrix_lr * batch_scale,
                name=f"muon_{'x'.join(str(dim) for dim in shape)}",
                momentum=muon_momentum,
                beta2=muon_beta2,
                weight_decay=weight_decay,
                ns_steps=muon_ns_steps,
            )
        )
    return OptimizerChain(optimizers)


def _adam_group(
    name: str, params: list[nn.Parameter], lr: float, *, weight_decay: float
) -> dict[str, Any]:
    return {
        "params": params,
        "lr": lr,
        "initial_lr": lr,
        "weight_decay": weight_decay,
        "name": name,
    }


def _polar_express(x: torch.Tensor, *, ns_steps: int) -> torch.Tensor:
    coeffs = [
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    ]
    if ns_steps != 5:
        raise ValueError("Only ns_steps=5 is supported")

    original_dtype = x.dtype
    x = x.to(torch.float32)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)

    if x.size(-2) >= x.size(-1):
        for a, b, c in coeffs:
            xx = x.mT @ x
            x = a * x + b * (x @ xx) + c * (x @ xx @ xx)
    else:
        for a, b, c in coeffs:
            xx = x @ x.mT
            x = a * x + b * (xx @ x) + c * (xx @ xx @ x)
    return x.to(original_dtype)


def _normuon_scale(
    update: torch.Tensor, second_buffer: torch.Tensor, *, beta2: float
) -> torch.Tensor:
    if second_buffer.shape[-1] == 1:
        v = update.square().mean(dim=1, keepdim=True)
        red_dim_size = update.size(1)
    else:
        v = update.square().mean(dim=0, keepdim=True)
        red_dim_size = update.size(0)

    second_buffer.lerp_(v, 1.0 - beta2)
    v_mean = second_buffer / (1.0 - beta2)
    old_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()

    scale = second_buffer.clamp_min(1e-10).rsqrt()
    scaled = update * scale
    new_norm = scaled.square().sum(dim=(-2, -1), keepdim=True).sqrt()
    return scaled * (old_norm / new_norm.clamp_min(1e-10))
