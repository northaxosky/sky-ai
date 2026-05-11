"""Generate text from a trained model checkpoint.

Loads the checkpoint saved by model.py and samples a stream of characters.
Each run produces a different sample because torch.multinomial is stochastic.
Run `uv run python model.py` first to produce the checkpoint, then re-run this
as many times as you want.

Usage:
    uv run python generate.py                  # 500 chars (default)
    uv run python generate.py --max-tokens 2000   # longer sample
"""
import argparse
import torch
from pathlib import Path

# Import the model class + device constant from model.py.
from model import BigramLanguageModel, device

CHECKPOINT_PATH = Path(__file__).parent / 'checkpoint.pt'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained model checkpoint."
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Number of new tokens to generate (default: 500).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CHECKPOINT_PATH}. "
            "Run `uv run python model.py` first to train and save one."
        )

    # Load the checkpoint. weights_only=False because the dict contains stoi/itos
    # (non-tensor Python objects) in addition to the state_dict.
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    vocab_size = ckpt['vocab_size']
    itos = ckpt['itos']

    decode = lambda l: ''.join(itos[i] for i in l)

    # Reconstruct the model and load its trained weights.
    model = BigramLanguageModel(vocab_size).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()  # disable dropout for inference

    # Sample.
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=args.max_tokens)[0].tolist()
    print(decode(generated))


if __name__ == '__main__':
    main()
