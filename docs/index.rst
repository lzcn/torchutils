Introduction
============
.. image:: https://readthedocs.org/projects/torchutils/badge/?version=latest
   :target: https://torchutils.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Essential PyTorch utilities: logging, checkpoints, config I/O, backbones, distributed training helpers.

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/lzcn/torchutils.git

Quick Start
-----------

.. code-block:: python

   import torchutils as tu

   # Logging
   tu.setup_logger(level="INFO", log_file="train.log")
   logger = tu.get_logger(__name__)

   # Config I/O
   config = tu.load_config("config.yaml")
   tu.save_config("output.json", config)

   # Backbones
   model, dim = tu.backbone("resnet50")

   # Device transfer
   batch = tu.to(batch, "cuda")

   # Checkpoints
   saver = tu.ModelSaver("checkpoints", n_saved=5, save_best=True)
   saver.save(model, score=0.95, epoch=10)
   tu.load_pretrained(model, saver.best_checkpoint)

   # Distributed training
   if tu.get_rank() == 0:
       print("Rank 0 only")

   @tu.rank_zero_only
   def save_results(data):
       pass  # Only executes on rank 0

Contents
--------

.. toctree::
   :maxdepth: 2

   api

License
-------

MIT License
