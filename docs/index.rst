torchutils
==========

A lightweight and modular PyTorch utility library designed for research and rapid prototyping.

.. image:: https://readthedocs.org/projects/torchutils/badge/?version=latest
   :target: https://torchutils.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
   :alt: Python Version

Features
--------

``torchutils`` provides essential utilities for PyTorch development:

**Core Utilities**
   - Device management and tensor operations
   - Model backbone loading with unified interface
   - Distributed training helpers

**I/O Operations**
   - Model checkpoint management with automatic versioning
   - File system utilities for scanning and validation
   - JSON/CSV serialization helpers

**Logging**
   - Rank-zero-only logging for distributed training
   - Flexible formatters and configuration
   - Integration with external frameworks (Hydra, etc.)

Installation
------------

Install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/lzcn/torchutils.git

Quick Start
-----------

.. code-block:: python

   import torch
   from torchutils import backbone, rank_zero_only, to
   from torchutils.logger import get_logger
   from torchutils.io import ModelSaver

   # Move data to device easily
   data = {"x": torch.randn(10, 3), "y": torch.randn(10, 1)}
   data = to(data, "cuda")

   # Load pretrained backbones
   model, out_dim = backbone("resnet50")

   # Setup logging for distributed training
   logger = get_logger(__name__)
   logger.info("Training started")

   # Save model checkpoints automatically
   saver = ModelSaver("./checkpoints", save_best=True)
   saver.save(model, accuracy_score, epoch)

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/io
   modules/logging
   modules/utilities

.. toctree::
   :maxdepth: 1
   :caption: Additional Info

   examples

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`