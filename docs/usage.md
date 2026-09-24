# torchutils usage guide

Conventions used throughout:

```python
import torchutils as tu
```

Everything works in plain single-process scripts. When `torch.distributed`
is initialized, the side-effecting entry points (logging, checkpoint
writes) automatically act on rank 0 only — you never need `if rank == 0:`
guards.

---

## `tu.setup_logger(...)`

```python
tu.setup_logger(
    level="INFO",          # default level for both handlers
    stream_level=None,     # console level (falls back to level)
    file_level=None,       # file level (falls back to level)
    log_file=None,         # path; no file logging if None
    file_mode="a",         # "a" or "w"
    format_string=None,    # custom logging format
    date_format=None,      # custom date format
)
```

Configures the **root logger** with a console handler and an optional
file handler. Behavior:

- Under distributed training only rank 0 emits records.
- Safe to call repeatedly — handlers are reset on every call, so it is
  fine to call it at the top of each entry-point script.
- A filter is attached to the handlers, so any logger in your project
  (`logging.getLogger(__name__)`) is deduplicated for free.
- Raises `ValueError` for an unknown level string (e.g. `"FLOOD"`).

```python
tu.setup_logger(log_file="runs/exp1/train.log")               # both INFO
tu.setup_logger(stream_level="DEBUG", file_level="WARNING")   # per-handler
logging.getLogger(__name__).info("hello")                     # just works
```

## `tu.to(data, device)`

```python
tu.to(data, device="cuda", non_blocking=True)
```

Recursively moves tensors to `device` through arbitrarily nested
dicts / lists / tuples. Rules:

- Namedtuples keep their type (not flattened to plain tuples).
- Non-tensor leaves (strings, ints, arrays of metadata) pass through.
- `non_blocking=True` enables async H2D/D2H copies when the source or
  staging memory is pinned; harmless otherwise.

```python
batch = {"img": imgs, "meta": [ids, (scale,)], "tag": "train"}
batch = tu.to(batch, "cuda")          # tensors moved, "tag" untouched
```

## `tu.ModelSaver(dirname, ...)`

```python
saver = tu.ModelSaver(
    dirname,                 # created if missing
    filename_prefix=None,    # optional "{prefix}_" for all filenames
    n_saved=5,               # how many best scored checkpoints to keep
    save_latest=False,       # rolling "{prefix}_latest.pt"
    save_best=True,          # "{prefix}_best.pt"
    mode="max",              # "max": higher score is better; "min": lower
)
saver.save(model_or_state_dict, score, epoch=None)
```

Keeps a top-`n_saved` leaderboard of checkpoints. Semantics:

- Scored files are named `{prefix}_{score:.4f}_epoch_{epoch}.pt`
  (`_epoch_N` omitted when `epoch=None`).
- `save` writes **only on rank 0** and is a no-op elsewhere; writes are
  atomic (temp file + `os.replace`), so a killed job never leaves a
  truncated checkpoint.
- The new file is written *before* the evicted one is deleted — a crash
  between the two steps never loses data.
- Saving an identical `(score, epoch)` again overwrites that file in
  place and does not double-book the leaderboard.
- `save` accepts a module or a bare state dict.

After saving:

```python
saver.best_checkpoint   # path to the best file, or None if never saved
saver.history           # [(score, path), ...] kept entries, worst first
```

Typical loop:

```python
saver = tu.ModelSaver("ckpt", n_saved=3, save_latest=True)
for epoch in range(epochs):
    ...
    saver.save(model, score=val_acc, epoch=epoch)   # latest always updated
```

Raises `ValueError` on `mode` not in `{"max", "min"}` or `n_saved < 1`.

## `tu.load_pretrained(net, path_or_state_dict)`

```python
tu.load_pretrained(
    net,
    path_or_state_dict,   # checkpoint path or a state dict
    weights_only=True,    # only tensors unpickled (safe against malicious files)
    strict=False,         # True: raise on ANY mismatch
) -> net
```

Loads weights into `net`, by default **skipping** missing and
shape-mismatched keys — made for fine-tuning across architecture drift
(updated backbone, replaced head). Behavior:

- Dicts containing a `"state_dict"` key (full training checkpoints like
  `{"state_dict": ..., "epoch": ..., "optimizer": ...}`) are unwrapped
  automatically.
- Files are loaded to CPU first, so checkpoints saved on GPU machines
  load anywhere.
- `strict=True` raises `RuntimeError` listing the counts of missing,
  unexpected, and shape-mismatched keys.

```python
tu.load_pretrained(model, saver.best_checkpoint)        # forgiving
tu.load_pretrained(model, "pretrained.pt")              # from file
tu.load_pretrained(model, state_dict, strict=True)      # hard-fail
```

## `tu.FeatureHook` / `tu.GradHook`

```python
with tu.FeatureHook(model, layers) as features:   # forward outputs
    ...
with tu.GradHook(model, layers) as grads:         # output grads after backward()
    ...
```

`layers` is one module name or a list, resolved with
`nn.Module.get_submodule` — dotted paths like `"encoder.blocks.3"` work.
The context manager yields the capture dict:

- Keys are the layer names you passed in; values are tensors.
- If a layer fires more than once, the **first** output is kept.
- `GradHook` stores the gradient of the module's first output tensor.
- Hooks are always removed on exit; an invalid layer name raises
  `AttributeError` **after** cleaning up the hooks registered so far.

```python
with tu.FeatureHook(model, ["layer2", "layer3"]) as features:
    output = model(x)
print(features["layer2"].shape)

with tu.GradHook(model, ["layer2"]) as grads:
    model(x).sum().backward()
print(grads["layer2"].abs().mean())   # quick "gradients are flowing" check
```

## `tu.rank_zero_only`

```python
@tu.rank_zero_only
def notify(msg): 
    requests.post(...)

notify("done")   # runs on rank 0, silently returns None elsewhere
```

On non-zero ranks the call is skipped and returns `None`.

## `tu.scan_files(path, suffix, recursive, relpath)`

```python
tu.scan_files(
    path="./",           # str or Path
    suffix=(),           # "jpg" or (".jpg", ".png"); leading dot optional
    recursive=False,     # descend into subdirectories
    relpath=False,       # return paths relative to path
) -> list[str]
```

- Hidden entries (dot-prefixed files/dirs like `.DS_Store`, `.git`) are
  always skipped; symlinked directories are not followed.
- Empty `suffix` means no filtering.
- Order follows the filesystem — sort it yourself if order matters.

```python
files = tu.scan_files("data/imagenet/", suffix=("jpg", "png"), recursive=True)
```

---

## Recipes

**Single-process or DDP training skeleton**

```python
import torchutils as tu

tu.setup_logger(log_file="runs/exp1/train.log")
saver = tu.ModelSaver("runs/exp1/ckpt", n_saved=3, save_latest=True)

for epoch in range(epochs):
    for batch in loader:
        batch = tu.to(batch, "cuda")
        loss = step(model, batch)
    score = validate(model)
    saver.save(model, score=score, epoch=epoch)   # rank-0 only, atomic
```

**Resume from the best epoch**

```python
tu.load_pretrained(model, saver.best_checkpoint)
```

**Fine-tune a model with a new head**

```python
model.head = NewHead(...)
tu.load_pretrained(model, "backbone.pt")   # head shapes differ: skipped
```

**Check that gradients reach early layers**

```python
with tu.GradHook(model, ["conv1", "layer4"]) as grads:
    loss(model(x), y).backward()
for name, g in grads.items():
    assert g is not None and torch.isfinite(g).all(), name
```

**Collect a dataset file list**

```python
files = tu.scan_files(root, suffix=("jpg", "png"), recursive=True)
```
