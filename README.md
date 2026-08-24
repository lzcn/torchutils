# torchutils

Minimal, general-purpose PyTorch utilities. One import, one-line calls.

## Installation

```bash
pip install git+https://github.com/lzcn/torchutils.git --upgrade
```

## Usage

```python
import torchutils as tu

# Logging (rank-0 only under distributed training)
tu.setup_logger(level="INFO", log_file="train.log")

# Device transfer for nested structures (dicts / lists / tuples of tensors)
batch = tu.to(batch, "cuda")

# Checkpoints: keeps the best 3 + a rolling latest copy, atomic writes, rank-0 only
saver = tu.ModelSaver("checkpoints", n_saved=3, save_latest=True)
saver.save(model, score=0.95, epoch=10)

# Load weights loosely: skips missing / shape-mismatched keys
tu.load_pretrained(model, saver.best_checkpoint)
tu.load_pretrained(model, "pretrained.pt")          # from file
tu.load_pretrained(model, state_dict, strict=True)  # raise on any mismatch

# Capture intermediate features / gradients by layer name
with tu.FeatureHook(model, ["layer2", "layer3"]) as features:
    output = model(x)

with tu.GradHook(model, ["layer2"]) as grads:
    output.sum().backward()

# Run a function on rank 0 only
@tu.rank_zero_only
def notify(): ...

# Dataset helpers (skips hidden files like .DS_Store)
files = tu.scan_files("data/", suffix=(".jpg", ".png"), recursive=True)
```

## License

MIT
