torchutils Documentation
========================

Personal library for PyTorch.

.. image:: https://readthedocs.org/projects/torchutils/badge/?version=latest
   :target: https://torchutils.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Philosophy
----------

``torchutils`` is designed with the following principles:

- ✅ **Minimal dependencies**: Only rely on PyTorch and Python standard libraries whenever possible.
- 🧩 **Modular and reusable**: Utility functions are simple, composable, and easy to integrate.
- 🧼 **Lightweight and clean**: Avoid unnecessary abstraction or complexity. Ideal for research, prototyping, or educational use.

This philosophy ensures that ``torchutils`` remains easy to maintain, portable across environments, and transparent for users.


Installation
------------

To install the latest version of ``torchutils``, you can use pip:

.. code-block:: bash

   pip install git+https://github.com/lzcn/torchutils.git --upgrade


Testing
-------

To run the tests:

.. code-block:: bash

   python -m unittest discover -s tests



Documentation
-------------

Hosted on `Read the Docs <https://torchutils.readthedocs.io/en/latest/>`_.


License
-------

MIT License: see the ``LICENSE`` file for details.

.. toctree::
   :maxdepth: 4
   :caption: 🔧 Modules

   torchutils
   torchutils.data
   torchutils.metrics
   torchutils.loss
   torchutils.plot
   torchutils.factory
   torchutils.files
   torchutils.logger
   torchutils.ops
   torchutils.ignite
   torchutils.io
   torchutils.param
   torchutils.ot
   torchutils.meter
   torchutils.layers

.. toctree::
   :hidden:

   genindex
   modindex
   search