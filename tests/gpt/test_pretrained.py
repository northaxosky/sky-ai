import pytest
import torch

from gpt.model import GPT

pytest = pytest.mark.slow  # downloads ~500MB, so run in CI


def test_from_pretrained_matches_hf_logits():
    from transformers import GPT2LMHeadModel

    ours = GPT.from_pretrained("gpt2").eval()
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    torch.manual_seed(0)
    idx = torch.randint(0, 50257, (1, 32))
    with torch.no_grad():
        our_logits, _ = ours(idx)
        hf_logits = hf(idx).logits

    max_diff = (our_logits - hf_logits).abs().max().item()
    assert max_diff < 1e-3, f"logits diverge: max abs diff = {max_diff}"
    assert torch.equal(our_logits.argmax(-1), hf_logits.argmax(-1))
