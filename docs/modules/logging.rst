Logging
=======

Rank-zero-only logging for distributed PyTorch training.

Quick Start
-----------

.. code-block:: python

   from torchutils.logger import config, get_logger
   
   # Configure once at startup
   config(level="INFO", log_file="app.log", formatter="simple")
   
   # Use throughout your code
   logger = get_logger(__name__)
   logger.info("Only rank 0 prints this")

Usage Patterns
--------------

**With manual configuration:**

.. code-block:: python

   config(level="INFO", formatter="simple")
   logger = get_logger(__name__)

**With external frameworks (Hydra, etc.):**

.. code-block:: python

   # No config() needed - framework handles it
   logger = get_logger(__name__)

**File and console logging:**

.. code-block:: python

   config(
       stream_level="INFO",
       file_level="DEBUG", 
       log_file="debug.log",
       stream_formatter="concise",
       file_formatter="default"
   )

Formatters
----------

================  ===============================================
Name              Format
================  ===============================================
``default``       ``[LEVEL] - MM-DD HH:MM:SS - [name.func:line]: message``
``simple``        ``[LEVEL] - MM-DD HH:MM:SS - [name]: message``
``concise``       ``MM-DD HH:MM:SS: message``
================  ===============================================

API Reference
-------------

.. autofunction:: torchutils.logger.get_logger

.. autofunction:: torchutils.logger.config

.. autofunction:: torchutils.logger.register_formatter