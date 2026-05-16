"""
Tokenize FineWeb-Edu and write sharded numpy files for training.

Default behavior reproduces Karpathy's build-nanogpt setup: 100 shards of
~100M tokens each, GPT-2 BPE, uint16 packing, first shard goes to val.

Usage:
    uv run python scripts/fineweb.py                    # full 10B run
    uv run python scripts/fineweb.py --max-shards 1     # local validation
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env from repo root (regardless of cwd)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Worker-side state. mp.Pool on Windows uses spawn, which re-imports this
# module in each worker, so module-level init runs per-worker. That's what
# we want for the encoder.
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def tokenize(doc: dict) -> np.ndarray:
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    arr = np.array(tokens)
    assert (arr >= 0).all() and (arr < 2**16).all(), "token id exceeds uint16 range"
    return arr.astype(np.uint16)


def write_shard(path: Path, tokens: np.ndarray) -> None:
    np.save(path, tokens)


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent.parent / "data" / "edu_fineweb10B"

    parser = argparse.ArgumentParser(
        description="Tokenize FineWeb-Edu and write sharded numpy files for training."
    )
    parser.add_argument(
        "--remote-name",
        default="sample-10BT",
        help="HF dataset config name (default: sample-10BT, the 10B-token sample)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=int(1e8),
        help="Tokens per shard (default: 100M)",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="Stop after N shards. 0 = no limit (default).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Where to write shard files (default: {default_output.relative_to(Path.cwd()) if default_output.is_relative_to(Path.cwd()) else default_output})",
    )
    return parser.parse_args()


def shard_path(output_dir: Path, shard_index: int) -> Path:
    split = "val" if shard_index == 0 else "train"
    return output_dir / f"edufineweb_{split}_{shard_index:06d}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=args.remote_name, split="train")
    nprocs = max(1, (os.cpu_count() or 2) // 2)

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        buffer = np.empty((args.shard_size,), dtype=np.uint16)
        token_count = 0
        progress: tqdm | None = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):  # pyright: ignore
            if token_count + len(tokens) < args.shard_size:
                buffer[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress is None:
                    progress = tqdm(
                        total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress.update(len(tokens))
                continue

            # Current shard is full. Fill the remainder, flush, start next shard
            # with the leftover from this document.
            remainder = args.shard_size - token_count
            buffer[token_count : token_count + remainder] = tokens[:remainder]
            if progress is not None:
                progress.update(remainder)
                progress.close()
                progress = None
            write_shard(shard_path(args.output_dir, shard_index), buffer)
            shard_index += 1

            if args.max_shards and shard_index >= args.max_shards:
                # Terminate explicitly before returning to avoid the Windows
                # mp.Pool teardown race ("concurrent send_bytes() calls").
                pool.terminate()
                pool.join()
                return

            leftover = tokens[remainder:]
            buffer[: len(leftover)] = leftover
            token_count = len(leftover)

        if token_count > 0:
            if progress is not None:
                progress.close()
            write_shard(shard_path(args.output_dir, shard_index), buffer[:token_count])


if __name__ == "__main__":
    main()
