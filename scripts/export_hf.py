"""Export a gpt2-family harness checkpoint to HuggingFace transformers (GPT2LMHeadModel) format.

The conversion is the exact inverse of gpt.model.GPT.from_pretrained: our state-dict keys
already match HF's, so we only transpose the four Conv1D matmuls, drop the training vocab
padding (50304 -> 50257), and let weight-tying handle lm_head. A logits-match check against
the harness model verifies correctness before anything is published.

    uv run python scripts/export_hf.py --checkpoint checkpoints/gpt2-muon/best.pt --out checkpoints/gpt2-muon-hf
    # then, with a write-scoped HF_TOKEN in .env:
    uv run python scripts/export_hf.py --checkpoint ... --out ... --repo <user>/gpt2-muon-124m --push
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from harness.checkpoint import load_checkpoint
from harness.training.loop import build_model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# HF GPT-2 stores these as Conv1D (in, out); our nn.Linear is (out, in) -> transpose on export.
_TRANSPOSE = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
_REAL_VOCAB = 50257  # gpt2 BPE; training pads to 50304
_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _to_hf_state_dict(model_state: dict) -> dict:
    hf: dict = {}
    for k, v in model_state.items():
        if k == "lm_head.weight":
            continue  # tied to wte; HF re-ties from transformer.wte.weight
        if k.endswith(_TRANSPOSE):
            v = v.t()
        if k == "transformer.wte.weight":
            v = v[:_REAL_VOCAB]
        hf[k] = v.contiguous().clone()
    return hf


def convert(checkpoint: Path, out: Path, dtype: str) -> None:
    bundle = load_checkpoint(checkpoint)
    mc = bundle.config.model
    if mc.family != "gpt2":
        raise SystemExit(f"export_hf only supports family=gpt2 (got {mc.family!r})")

    cfg = GPT2Config(
        vocab_size=_REAL_VOCAB,
        n_positions=mc.block_size,
        n_embd=mc.n_embd,
        n_layer=mc.n_layer,
        n_head=mc.n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        # defaults already match ours: activation_function="gelu_new", layer_norm_epsilon=1e-5, tie_word_embeddings=True
    )
    hf_model = GPT2LMHeadModel(cfg)
    missing, unexpected = hf_model.load_state_dict(
        _to_hf_state_dict(bundle.model_state), strict=False
    )
    if unexpected:
        raise SystemExit(f"unexpected keys (conversion bug): {unexpected}")
    print(f"missing (expected: tied lm_head + causal-mask buffers): {sorted(set(missing))[:4]}...")
    hf_model.tie_weights()
    hf_model = hf_model.to(_DTYPES[dtype]).eval()

    out.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(out)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(out)
    print(f"wrote HF model -> {out} (dtype={dtype})")


def verify(checkpoint: Path, out: Path) -> None:
    bundle = load_checkpoint(checkpoint)
    ours = build_model(bundle.config.model)
    ours.load_state_dict(bundle.model_state)
    ours.eval()
    hf = AutoModelForCausalLM.from_pretrained(out).eval()

    torch.manual_seed(0)
    idx = torch.randint(0, _REAL_VOCAB, (1, 32))
    with torch.no_grad():
        ours_logits, _ = ours(idx)
        hf_logits = hf(idx).logits
    diff = (ours_logits[..., :_REAL_VOCAB].float() - hf_logits.float()).abs().max().item()
    print(f"verify: max|delta logits| over first {_REAL_VOCAB} vocab = {diff:.3e}")
    if diff > 1e-2:
        raise SystemExit(f"verify FAILED: logits diverge ({diff:.3e}) -- conversion is wrong")
    print("verify OK")


def push(out: Path, repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="model")
    print(f"pushed -> https://huggingface.co/{repo}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", choices=list(_DTYPES), default="fp32")
    ap.add_argument("--repo", type=str, default=None)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    convert(args.checkpoint, args.out, args.dtype)
    verify(args.checkpoint, args.out)
    if args.push:
        if not args.repo:
            raise SystemExit("--push requires --repo <user>/<name>")
        push(args.out, args.repo)


if __name__ == "__main__":
    main()
