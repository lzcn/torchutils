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

🚀 **Core Utilities**
   - Device management and tensor operations
   - Model backbone loading and management  
   - Distributed training helpers

📊 **Data & I/O**
   - LMDB and file readers for large datasets
   - Model saving and loading utilities
   - JSON/CSV serialization helpers

📈 **Metrics & Logging**
   - Colorized logging with flexible formatting
   - Performance metrics and meters
   - Optional transport (OT) algorithms

Installation
------------

Install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/lzcn/torchutils.git

Quick Start
-----------

.. code-block:: python

   import torch
   from torchutils.ops import to
   from torchutils.backbones import backbone
   from torchutils.logger import get_logger

   # Move data to device easily
   data = {"x": torch.randn(10, 3), "y": torch.randn(10, 1)}
   data = to(data, "cuda")

   # Load pretrained backbones
   model, out_dim = backbone("resnet50", weights="IMAGENET1K_V1")

   # Get colored logger
   logger = get_logger("my_app")
   logger.info("Training started")

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: 🔧 Core Modules

   modules/core
   modules/io
   modules/utils

.. toctree::
   :maxdepth: 1
   :caption: 📚 Additional Info

   installation
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`