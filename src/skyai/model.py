from __future__ import annotations

import inspect
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
from dotenv import load_dotenv
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from skyai.eval.hellaswag import get_most_likely_row, iterate_examples, render_example

# Load .env from repo root (regardless of cwd)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ===============================


@dataclass
class GPTConfig:
    block_size: int = 1024  # Maximum Sequence Length
    vocab_size: int = 50257  # Number of Tokens: 50k BPE merges + tokens
    n_layer: int = 12  # Number of Layers
    n_head: int = 12  # Number of Heads
    n_embed: int = 768  # Embedding Dimension


class CausalSelfAttention(nn.Module):
    config: GPTConfig
    c_attn: nn.Linear
    c_proj: nn.Linear

    n_head: int
    n_embed: int
    bias: torch.Tensor

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # Key, Query, Value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)

        # Output Projection and regularization
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # pyright: ignore
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # Mask/Bias following the OpenAI/HF naming
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the batch size, sequence length, & embedding dimensionality
        B, T, C = x.size()

        # Calculate query, key, & values for all heads in batch and move head forward
        # nh: # of heads, hs: head size, C: number of channels (nh * ns)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Attention: materialize the large (T, T) matrix for all the queries and keys
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    config: GPTConfig
    c_fc: nn.Linear
    gelu: nn.GELU
    c_proj: nn.Linear

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # pyright: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    config: GPTConfig
    ln_1: nn.LayerNorm
    attn: CausalSelfAttention
    ln_2: nn.LayerNorm
    mlp: MLP

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    config: GPTConfig
    transformer: nn.ModuleDict
    lm_head: nn.Linear

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Follow Hugging Face schema so we can load it easily
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.block_size, config.n_embed),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight Sharing Scheme
        self.transformer.wte.weight = self.lm_head.weight  # pyright: ignore

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        wte = cast(nn.Embedding, self.transformer.wte)
        wpe = cast(nn.Embedding, self.transformer.wpe)
        blocks = cast(nn.ModuleList, self.transformer.h)
        ln_f = cast(nn.LayerNorm, self.transformer.ln_f)

        # idx is of shape (B, T)
        _, T = idx.size()
        assert self.config.block_size >= T, (
            f"Cannot forward sequence of length{T}, block size invalid"
        )

        # Forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = wpe(pos)
        tok_emb = wte(idx)
        x = tok_emb + pos_emb

        # Forward the blocks of the transformer
        for block in blocks:
            x = block(x)

        # Forward the final layernorm and the classifier
        x = ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str) -> GPT:
        """Loads pre-trained GPT-2 model weights from Hugging Face"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        # Rank 0 or non-DDP: print once. In DDP, suppress noise from other ranks.
        if os.environ.get("RANK", "0") == "0":
            print(f"Loading weights from pre-trained GPT: {model_type}")

        # n_layer, n_head, n_embed are determined from model type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),  # 124M Parameters
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),  # 350M Parameters
            "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),  # 774M Parameters
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M Parameters
        }[model_type]

        config_args["vocab_size"] = 50257  # Always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # Always 1024 for GPT model checkpoints

        # Create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]  # Discard

        # Initialize a hugging face/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # Transpose the weights (OpenAI checkpoints with "Conv1D" module)
        assert len(sd_keys_hf) == len(sd_keys), (
            f"Mismatched Keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy of the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups where 2D parameters will be weight delayed
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if os.environ.get("RANK", "0") == "0":
            print(
                f"# Decayed Parameters Tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"# Non-Decayed Parameter Tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


# ============================================
def load_tokens(filename):
    npt = np.load(filename)
    # Defensive cast: shard .npy files are written as uint16, but be explicit
    # about the int32 intermediate before going to torch.long. Karpathy added
    # this post-video to avoid edge cases with unexpected source dtypes.
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int, split: str):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # Get the shard filenames
        data_root = Path(__file__).resolve().parent.parent.parent / "data" / "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split: {split}."
        if master_process:
            print(f"Found {len(shards)} shards for split: {split}")
        self.reset()

    def reset(self):
        # State, init at shard zero. Position must be per-rank offset, not
        # num_processes - otherwise all DDP ranks read the same data and you lose
        # all data parallelism (silently, gradients still sync).
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + (B * T) + 1]

        # Inputs & Targets
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes

        # If loading the next batch would be out of bounds: advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    def state_dict(self) -> dict:
        return {"current_shard": self.current_shard, "current_position": self.current_position}

    def load_state_dict(self, state: dict) -> None:
        # Only the saving rank's position is exact; other ranks restart at their
        # rank-offset within the same shard. Costs at most one microbatch of skew.
        self.current_shard = state["current_shard"]
        self.tokens = load_tokens(self.shards[self.current_shard])
        if self.process_rank == 0:
            self.current_position = state["current_position"]
        else:
            self.current_position = self.B * self.T * self.process_rank


# ============================================
# Set up distributed data parallel
ddp = int(os.environ.get("RANK", -1)) != -1

# torchrun --standalone --nproc_per_node=x model.py
if ddp:
    # Use of DDP requires CUDA
    assert torch.cuda.is_available(), "DDP requires CUDA/GPU"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    print(f"Using Distributed Data Parallel w/ #{ddp_world_size}")
else:
    # Non ddp run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # Attempt to auto detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Non-DDP using device: {device}")

# torch.autocast wants device_type ("cuda" / "cpu" / "mps"), not a full device
# string like "cuda:0". PyTorch is strict about this distinction.
device_type = "cuda" if device.startswith("cuda") else device

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288

# Micro Batch Size & Sequence Length
# B=64 matches Karpathy reference. Requires ~25GB VRAM per GPU (40GB+ recommended).
# For local dev on smaller GPUs (e.g., 4090 24GB), drop to B=16.
B = 64
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, (
    "Make sure total_batch_size is divisible by B * T"
)
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"> Calculated gradient accumulation steps: {grad_accum_steps}")

# Train/Data loader
train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
)
val_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
)
torch.set_float32_matmul_precision("high")

# Create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained('gpt2-xl')
model.to(device)

# torch.compile can introduce subtle BF16 numerical drift vs eager mode.
# Karpathy's reference disables it by default. Set SKYAI_COMPILE=1 to enable.
use_compile = sys.platform == "linux" and os.environ.get("SKYAI_COMPILE", "0") == "1"
if use_compile:
    model = torch.compile(model)
    if master_process:
        print("Using torch-compiled GPT")
elif master_process:
    print("Running uncompiled (eager mode)")
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
# raw_model unwraps both DDP and torch.compile so we can call methods like
# configure_optimizers, and use eager-mode forward for sampling (where compile's
# changing-shape recompiles would dominate runtime).
raw_model = model.module if ddp else model
raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model  # pyright: ignore

# Cosine decay learning rate (GPT-3)
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073


def get_lr(it):
    # Linear warmup for warmup_iters steps, min learning rate when past max steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr

    # Use cosine decay down to minimum learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)  # pyright: ignore
enc = tiktoken.get_encoding("gpt2")

# Logs and checkpoints (separate directories; logs is small text, checkpoints is large binaries)
log_dir = "logs"
checkpoint_dir = "checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

# Resume from latest checkpoint if any. Auto-detect; flip RESUME=False to force fresh start.
RESUME = True
start_step = 0
val_loss_accum: torch.Tensor | float | None = None
wandb_run_id: str | None = None

if RESUME:
    candidates = sorted(Path(checkpoint_dir).glob("model_*.pt"))
    if candidates:
        latest = candidates[-1]
        if master_process:
            print(f"Resuming from {latest}")
        try:
            ckpt = torch.load(latest, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {latest}: {e}") from e

        # Config mismatch is a warning, not an error - allows architecture iteration on old checkpoints
        ckpt_config = ckpt.get("config", {})
        current_config = asdict(raw_model.config)  # pyright: ignore
        if ckpt_config != current_config and master_process:
            print(
                f"WARNING: checkpoint config differs from current. Loading anyway. "
                f"ckpt={ckpt_config} current={current_config}"
            )

        raw_model.load_state_dict(ckpt["model"])  # pyright: ignore

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        elif master_process:
            print("WARNING: no optimizer state in checkpoint; starting optimizer fresh")

        if "train_loader" in ckpt:
            train_loader.load_state_dict(ckpt["train_loader"])
        elif master_process:
            print(
                "WARNING: no data loader state in checkpoint; data loader will restart from shard 0"
            )

        start_step = ckpt["step"] + 1
        val_loss_accum = ckpt.get("val_loss")
        wandb_run_id = ckpt.get("wandb_run_id")
        if master_process:
            print(f"Resumed at step {start_step}, val_loss={val_loss_accum}")

# Truncate log only on fresh start; append on resume to preserve history
log_mode = "a" if start_step > 0 else "w"
with open(log_file, log_mode) as file:
    if start_step > 0:
        file.write(f"\n--- Resumed from step {start_step} ---\n")

# wandb: master process only, resumes the same run if a run id was loaded from checkpoint
if master_process:
    if wandb_run_id is None:
        wandb_run_id = wandb.util.generate_id()  # pyright: ignore[reportAttributeAccessIssue]
    wandb.init(
        project="skyai",
        id=wandb_run_id,
        resume="allow",
        config=asdict(raw_model.config)  # pyright: ignore
        | {  # pyright: ignore
            "max_steps": max_steps,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "total_batch_size": total_batch_size,
            "micro_batch_size": B,
            "sequence_length": T,
            "grad_accum_steps": grad_accum_steps,
            "ddp_world_size": ddp_world_size,
            "weight_decay": 0.1,
        },
    )

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    # Evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()

        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20

            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation Loss: {val_loss_accum:.4f}")
            with open(log_file, "a") as file:
                file.write(f"Step: {step}, Validation Loss: {val_loss_accum:.4f}\n")
            wandb.log({"val/loss": float(val_loss_accum)}, step=step)  # pyright: ignore

    # Evaluate hellaswag
    if step % 250 == 0 or last_step:
        model.eval()
        num_correct_norm = 0
        num_total = 0

        for i, example in enumerate(iterate_examples("val")):
            if i % ddp_world_size != ddp_rank:
                continue

            # Render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # Get the logits (raw_model: avoid torch.compile recompile per varying example length)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = raw_model(tokens)  # pyright: ignore
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # Reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)

            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)

            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total

        # Log the results
        if master_process:
            print(f"HellaSwag Accuracy: {num_correct_norm} / {num_total} = {acc_norm:.4f}")
            with open(log_file, "a") as file:
                file.write(f"Step: {step}, HellaSwag: {acc_norm:.4f}\n")
            wandb.log({"eval/hellaswag_acc": acc_norm}, step=step)

    # Evaluate sampling of the model
    if (step % 250 == 0 and step != 0) or last_step:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        x = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)

        while x.size(1) < max_length:
            # Forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = raw_model(x)  # pyright: ignore

                # Take the logits at the last position & get the probabilities
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)

                # do top-k sampling of 50 & select a token
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)

                # Gather the corresponding indices & append to the sequence
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat((x, xcol), dim=1)

        # Print the generated text
        sample_texts: list[str] = []
        for i in range(num_return_sequences):
            tokens = x[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"Rank: {ddp_rank}, Sample: {i + 1} > {decoded}")
            sample_texts.append(decoded)
        if master_process:
            wandb.log(
                {"samples": wandb.Table(columns=["text"], data=[[t] for t in sample_texts])},
                step=step,
            )

    # Training Loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # require_backward_grad_sync is read by BOTH forward and backward DDP hooks,
        # so it must be set BEFORE the forward pass, not just before backward.
        # See karpathy/build-nanogpt README errata for the original bug.
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1  # pyright: ignore

        # Use bfloat16 for optimal speed/precision
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # Scale the loss to account for gradient accumulation
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()

    # Track statistics
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_sec = tokens_processed / (dt / 1000)
    if master_process:
        print(
            f"Step {step} | LR: {lr:.4e} | Loss: {loss_accum:.6f} | Norm: {norm:.4f} | dT: {dt:.2f}ms | Tok/sec: {tokens_sec:.2f}"
        )
        with open(log_file, "a") as file:
            file.write(f"Step: {step}, Training Loss: {loss_accum:.6f}\n")
        wandb.log(
            {
                "train/loss": float(loss_accum),
                "train/lr": lr,
                "train/grad_norm": float(norm),
                "train/tokens_per_sec": tokens_sec,
                "train/dt_ms": dt,
            },
            step=step,
        )
        if step > 0 and (step % 10000 == 0 or last_step):
            checkpoint_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
            checkpoint = {
                "model": raw_model.state_dict(),  # pyright: ignore
                "config": asdict(raw_model.config),  # pyright: ignore
                "step": step,
                "val_loss": val_loss_accum,
                "optimizer": optimizer.state_dict(),
                "train_loader": train_loader.state_dict(),
                "wandb_run_id": wandb_run_id,
            }
            torch.save(checkpoint, checkpoint_path)

if master_process:
    wandb.finish()
if ddp:
    destroy_process_group()
