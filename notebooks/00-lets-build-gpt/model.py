"""Character-level GPT trained on Tiny Shakespeare.

Following Karpathy's "Let's build GPT" (Zero to Hero, video 7). Defines the model
classes (Head, MultiHeadAttention, FeedForward, Block, BigramLanguageModel) and,
when run directly, trains them on Shakespeare and saves a checkpoint for `generate.py`
to load. Importing this module from elsewhere skips training but exposes the classes
and helpers for reuse.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# === Hyperparameters ===
seed = 1337             # For reproducibility
batch_size = 64         # How many independent sequences we process in parallel per training step
block_size = 256        # Maximum context length: the model sees the last N tokens when predicting
max_iters = 5000        # Total training steps
eval_interval = 500     # Steps between train/val loss evaluations during training
learning_rate = 3e-4    # AdamW step size; 3e-4 is the canonical "Karpathy constant" for transformers
eval_iters = 200        # Mini-batches averaged per loss estimate (more = smoother numbers, slower eval)
n_embed = 384           # Embedding dimension; size of the vectors flowing through the network
n_head = 6              # Parallel attention heads per block; head_size = n_embed // n_head = 64
n_layer = 6             # Number of transformer blocks stacked (depth of the network)
dropout = 0.2           # Fraction of activations zeroed during training; disabled in eval mode
device = 'cuda'         # 'cuda' for GPU, 'cpu' as fallback

# === Paths (resolved relative to this file, not the cwd) ===
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = (SCRIPT_DIR / '..' / '..' / 'data' / 'shakespeare.txt').resolve()
CHECKPOINT_PATH = SCRIPT_DIR / 'checkpoint.pt'

torch.manual_seed(seed)

# === Data ===
with open(DATA_PATH, 'r', encoding='utf-8') as file:
    text = file.read()

# the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mapping of characters to integers
stoi = { c : i for i, c in enumerate(chars) }
itos = { i : c for i, c in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Encoder: take a string, output list of integers
decode = lambda l: ''.join(itos[i] for i in l) # Decoder: Take a list of ints, output string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs: x, targets: y
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# One Head of Self-Attention
class Head(nn.Module):
    tril: torch.Tensor  # For Pylance Type Checker

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)

        # Compute attention scores ("affinities")
        weights = query @ key.transpose(-2, -1) * C ** -0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # Perform weighted aggregation of values
        value = self.value(x)
        out = weights @ value
        return out
    

# Multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

# Simple linear layer followed by a non-linearity
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Transformer Block: communication followed by computation
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: # of heads
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Simple Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(* [Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # Final Layer Normalization
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        position_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block size tokens
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
        
        return idx

if __name__ == '__main__':
    model = BigramLanguageModel(vocab_size)
    model = model.to(device)
    print(f'Model on: {next(model.parameters()).device}') # Should print CUDA:0 if on GPU

    # Create PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # Occassionally evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f'Step {iter}: Train Loss - {losses['train']:.4f}, Val Loss - {losses['val']:.4f}')

        # Sample a batch of data
        xb, yb = get_batch('train')

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save a checkpoint (script-relative path so it lands next to model.py)
    torch.save({
        'model_state': model.state_dict(),
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
    }, CHECKPOINT_PATH)
    print(f'Checkpoint saved to {CHECKPOINT_PATH}')

    # Generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

''' Output:
Model on: cuda:0
Step 0: Train Loss - 4.2849, Val Loss - 4.2823
Step 500: Train Loss - 2.0016, Val Loss - 2.0883
Step 1000: Train Loss - 1.5951, Val Loss - 1.7720
Step 1500: Train Loss - 1.4386, Val Loss - 1.6373
Step 2000: Train Loss - 1.3414, Val Loss - 1.5734
Step 2500: Train Loss - 1.2787, Val Loss - 1.5317
Step 3000: Train Loss - 1.2269, Val Loss - 1.5094
Step 3500: Train Loss - 1.1820, Val Loss - 1.4866
Step 4000: Train Loss - 1.1452, Val Loss - 1.4848
Step 4500: Train Loss - 1.1107, Val Loss - 1.4823


SLY:
Sir.

KING RICHARD III:
He so, Jove's queen, to Mina.

QUEEN Northan's hearth, sir; for thee, oxford daughter,
kneel 'tis; he so nothing it; 'tis well no
but issue, as I call thee, he will hang.

KING RICHARD II:
I am so; for a country, and saver'd fild.
Rance thou bellow'st thy son, by my fellow?

RATCLIFF:
Hang Richard, this Jove is that flatter.
But, foolish thy life and by the wood--grad-vingiling
fulled by crosbrack of my loyal king,
That whose high an ear-bearancer hence thee
I in th
'''