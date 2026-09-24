# torchutils

Tiny, general-purpose utilities for PyTorch training. One import, one-line calls.

## Why

Every training script reinvents the same small machinery: rank-guarded logging,
recursive `batch.to(device)`, checkpoint bookkeeping that tracks the best epochs,
and weight loading that doesn't explode over a missing key. `torchutils` wraps
that machinery into a handful of small, tested primitives:

- **DDP-aware** — logging and checkpoint writes are rank-0 only out of the box,
  so no `if rank == 0:` scattered through your code.
- **Crash-safe** — checkpoints are written atomically (temp file + rename), a
  killed job never leaves a half-written file behind.
- **Forgiving by default** — `load_pretrained` skips missing or shape-mismatched
  keys instead of raising; pass `strict=True` when you want the opposite.
- **Small and typed** — the only dependency is `torch` itself, and the package
  ships a `py.typed` marker (PEP 561) so type checkers see the annotations.

Requires Python ≥ 3.10.

## Installation

```bash
pip install git+https://github.com/lzcn/torchutils.git --upgrade
```

## Quick start

The whole library in the shape of a training loop:

```python
import torchutils as tu

tu.setup_logger(log_file="runs/exp1/train.log")   # console + file, rank-0 only
saver = tu.ModelSaver("runs/exp1/ckpt", n_saved=3, save_latest=True)

for epoch in range(epochs):
    for batch in loader:                          # batch may be arbitrarily nested
        batch = tu.to(batch, "cuda")              # dicts / lists / tuples of tensors

        # capture intermediate features for analysis or auxiliary losses
        with tu.FeatureHook(model, ["layer2"]) as features:
            loss = train_step(model, batch)

    saver.save(model, score=validate(model), epoch=epoch)   # keeps top-3 + latest

# later: resume from the best epoch, tolerating a changed head
tu.load_pretrained(model, saver.best_checkpoint)
```

A more detailed usage guide — full parameters, behaviors, and recipes —
lives in [docs/usage.md](docs/usage.md).

## API

### Logging — `tu.setup_logger(...)`

Configures the root logger with a console handler and an optional file handler.
Under distributed training only rank 0 emits records; safe to call again at any
time (handlers are reset on each call).

```python
tu.setup_logger(level="INFO", log_file="train.log", file_mode="w")
tu.setup_logger(stream_level="DEBUG", file_level="WARNING")   # per-handler levels
```

### Device transfer — `tu.to(data, device)`

Recursively moves tensors through nested dicts, lists, and tuples. Namedtuples
keep their type, and non-tensor leaves pass through untouched.

```python
batch = tu.to({"img": imgs, "meta": [ids, (scale,)]}, "cuda")
```

### Checkpointing — `tu.ModelSaver(dirname, ...)`

Keeps the best `n_saved` checkpoints by score, plus optional rolling ``latest``
and ``best`` copies. Writes are atomic and rank-0 only; filenames look like
`{prefix}_{score:.4f}_epoch_{epoch}.pt`.

```python
saver = tu.ModelSaver("ckpt", n_saved=3, save_latest=True, mode="min")
saver.save(model, score=val_loss, epoch=epoch)
saver.save(model.state_dict(), score=val_loss)   # a bare state dict works too

saver.best_checkpoint                            # path to the best file
```

### Weight loading — `tu.load_pretrained(net, path_or_state_dict)`

Loads weights, silently skipping missing and shape-mismatched keys — handy when
a backbone was updated or a head was replaced. Set `strict=True` to raise on any
mismatch instead. Full training checkpoints of the form
`{"state_dict": ..., "epoch": ...}` are unwrapped automatically.

```python
tu.load_pretrained(model, "pretrained.pt")
tu.load_pretrained(model, state_dict, strict=True)   # hard-fail on any mismatch
```

### Hooks — `tu.FeatureHook` / `tu.GradHook`

Context managers that capture forward outputs / output gradients of named
submodules. Hooks are always removed on exit — even if a layer name is wrong.

```python
with tu.FeatureHook(model, ["layer2", "layer3"]) as features:
    output = model(x)

with tu.GradHook(model, ["layer2"]) as grads:
    output.sum().backward()
```

### Misc

```python
# run a function on rank 0 only (no-op elsewhere)
@tu.rank_zero_only
def notify(): ...

# list dataset files; hidden files skipped, "jpg" and ".jpg" both work
files = tu.scan_files("data/", suffix=(".jpg", ".png"), recursive=True)
```

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

## License

MIT
