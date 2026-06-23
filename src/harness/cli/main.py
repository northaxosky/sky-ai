"""skyai CLI entry point"""

from __future__ import annotations

import os
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated

import torch
import typer

from harness.checkpoint import load_checkpoint
from harness.config.loader import load_config
from harness.config.schema import LogConfig, RunConfig
from harness.log import get_logger, setup_logging
from harness.sample import sample as sample_fn
from harness.training.loop import build_gpt_config
from skyai.model import GPT

app = typer.Typer(
    name="skyai",
    help="SkyAI training/eval/sample harness",
    no_args_is_help=True,
)

logger = get_logger(__name__)


def _rank_from_env() -> int:
    """Read torchrun-provided RANK env var, default to 0 for single-process runs"""
    return int(os.environ.get("RANK", "0"))


def _setup_run(cfg_path: Path, overrides: list[str], *, log_name: str) -> RunConfig:
    """Load config + overrides, init rank-aware logging, return validated RunConfig"""
    cfg = load_config(cfg_path, overrides)
    rank = _rank_from_env()
    cfg.log.dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.log, rank=rank, log_path=cfg.log.dir / log_name)
    return cfg


@app.callback()
def _root() -> None:
    """SkyAI training/eval/sample harness"""


@app.command()
def version() -> None:
    """SkyAI training/eval/sample harness"""
    typer.echo(_pkg_version("skyai"))


@app.command()
def train(
    config: Annotated[Path, typer.Option(help="Path to YAML config")],
    override: Annotated[
        list[str] | None,
        typer.Option(help="Override config field, e.g. --override model.n_layer=4"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(help="Auto-resume from latest checkpoint in cfg.checkpoint.dir"),
    ] = False,
) -> None:
    from harness.training.loop import train as run_train

    cfg = _setup_run(config, override or [], log_name="train.log")
    logger.info(f"train: {config=}, {resume=}, {cfg.total_batch_size=}, {cfg.schedule.max_steps=}")
    run_train(cfg, resume=resume)


@app.command(name="eval")
def evaluate(
    config: Annotated[Path, typer.Option(help="Path to YAML config")],
    checkpoint: Annotated[Path, typer.Option(help="Checkpoint path (.pt, .json, dir)")],
    override: Annotated[
        list[str] | None,
        typer.Option(help="Override config field, e.g. --override eval.hellaswag=true"),
    ] = None,
) -> None:
    """Run the eval suite on a checkpoint and print per metric results"""
    import tiktoken

    from harness.eval import run_evals

    cfg = _setup_run(config, override or [], log_name="eval.log")
    bundle = load_checkpoint(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc = bundle.config.model
    model = GPT(build_gpt_config(mc))
    model.load_state_dict(bundle.model_state)
    model.to(device).eval()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[cfg.dtype]
    enc = tiktoken.get_encoding(bundle.config.model.tokenizer)

    logger.info(f"eval: ckpt step={bundle.step}, {device=}, {cfg.dtype=}, evals={cfg.eval.evals}")
    results = run_evals(
        cfg.eval.evals, model, encoder=enc, device=device, rank=0, world_size=1, dtype=dtype
    )  # pyright: ignore

    for name, result in results.items():
        for metric, value in result.metrics.items():
            typer.echo(f"{name}/{metric}: {value:.4f} (n={result.num_examples})")


@app.command()
def sample(
    checkpoint: Annotated[Path, typer.Option(help="Checkpoint path (.pt, .json, dir)")],
    prompt: Annotated[str, typer.Option(help="Prompt text")] = "Hello, I'm a language model,",
    num_samples: Annotated[int, typer.Option(help="Number of completions to generate")] = 1,
    max_new_tokens: Annotated[int, typer.Option(help="Tokens to generate beyond the prompt")] = 50,
    temperature: Annotated[float, typer.Option(help="Sampling temperature (>0)")] = 1.0,
    top_k: Annotated[int, typer.Option(help="Top-k cutoff; 0 disables")] = 50,
    seed: Annotated[int | None, typer.Option(help="RNG seed for reproducibility")] = None,
    device: Annotated[str, typer.Option(help="auto, cuda, cuda:N, or cpu")] = "auto",
) -> None:
    """Generate text from a trained checkpoint"""
    import tiktoken

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = load_checkpoint(checkpoint)
    mc = bundle.config.model
    model = GPT(build_gpt_config(mc))
    model.load_state_dict(bundle.model_state)
    model.to(device).eval()

    enc = tiktoken.get_encoding(mc.tokenizer)

    rng: torch.Generator | None = None
    if seed is not None:
        rng = torch.Generator(device=device).manual_seed(seed)

    prompt_len = len(enc.encode(prompt))
    completions = sample_fn(
        model,
        enc,
        prompt,
        n_samples=num_samples,
        max_length=prompt_len + max_new_tokens,
        device=device,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        generator=rng,
        max_context_len=mc.block_size,
    )

    for i, text in enumerate(completions):
        if num_samples > 1:
            typer.echo(f"--- sample {i + 1}/{num_samples} ---")
        typer.echo(text)


@app.command()
def doctor(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional config to also check data/checkpoint paths"),
    ] = None,
) -> None:
    """Environment + project sanity checks"""
    from harness.cli.doctor import run_doctor

    raise typer.Exit(run_doctor(config_path=config))


@app.command()
def ablate(
    spec: Annotated[Path, typer.Option(help="Path to ablation YAML spec")],
    output_dir: Annotated[
        Path, typer.Option(help="Where per-variant runs and results.{md,json} land")
    ],
    dry_run: Annotated[bool, typer.Option(help="Print the variant plan, don't train")] = False,
    force: Annotated[bool, typer.Option(help="Re-run variants even if result.json exists")] = False,
) -> None:
    """Sequential parameter sweep: train each variant, write results.{md,json}"""
    from harness.ablation import run_ablation

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        LogConfig(dir=output_dir),
        rank=_rank_from_env(),
        log_path=output_dir / "ablation.log",
    )
    logger.info(f"ablate: {spec=}, {output_dir=}, {dry_run=}, {force=}")
    run_ablation(spec, output_dir, dry_run=dry_run, force=force)


if __name__ == "__main__":
    app()
