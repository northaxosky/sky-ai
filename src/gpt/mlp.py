import torch
import torch.nn as nn

from gpt.model import GPTConfig


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # expand: C -> 4C
        self.gelu = nn.GELU(approximate="tanh")  # OpenAI's exact non-linearity
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # contract: 4C -> C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
