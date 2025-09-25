Utilities
=========

Common utilities and helper functions exposed at the top level of ``torchutils``.

Formatting
----------

Structured display helpers for nested dictionaries and lists.

.. autofunction:: torchutils.format_display

Checkpoint Utilities
--------------------

Load checkpoints with flexible key-matching and helpful diagnostics.

.. autofunction:: torchutils.load_pretrained

.. autofunction:: torchutils.update_npz

Tensor Operations
-----------------

Device-aware helpers for moving nested structures.

.. autofunction:: torchutils.ops.to

.. autofunction:: torchutils.one_hot

Training Helpers
----------------

Utilities for loss aggregation, optimizer initialisation and device planning.

.. autofunction:: torchutils.infer_parallel_device

.. autofunction:: torchutils.gather_loss

.. autofunction:: torchutils.gather_mean

.. autofunction:: torchutils.init_optimizer

Model Backbones
---------------

Factory interface for common torchvision backbones.

.. autofunction:: torchutils.backbones.backbone

Distributed Training
--------------------

Utilities that only run on rank zero in multi-process setups.

.. autofunction:: torchutils.distributed.rank_zero_only

Configuration
-------------

Extended YAML loader with ``!include`` support.

.. autofunction:: torchutils.from_yaml

Singleton Pattern
-----------------

Lightweight singleton decorator for module-level resources.

.. autofunction:: torchutils.singleton.singleton
