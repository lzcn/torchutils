# torchutils

[![Documentation Status](https://readthedocs.org/projects/torchutils/badge/?version=latest)](https://torchutils.readthedocs.io/en/latest/?badge=latest)

Essential PyTorch utilities: logging, checkpoints, config I/O, backbones, distributed training helpers.

## Installation

```bash
pip install git+https://github.com/lzcn/torchutils.git --upgrade
```

## Usage

```python
import torchutils as tu

tu.setup_logger(level="INFO", log_file="train.log")
logger = tu.get_logger(__name__)

config = tu.load_config("config.yaml")
model, dim = tu.backbone("resnet50")
batch = tu.to(batch, "cuda")

saver = tu.ModelSaver("checkpoints", n_saved=5)
saver.save(model, score=0.95, epoch=10)
```

## License

MIT
