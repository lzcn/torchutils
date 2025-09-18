Installation
============

Requirements
------------

- Python 3.8+
- PyTorch 1.8+
- NumPy
- SciPy (for optimal transport functions)

Basic Installation
------------------

Install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/lzcn/torchutils.git

Development Installation
------------------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/lzcn/torchutils.git
   cd torchutils
   pip install -e .

Optional Dependencies
---------------------

Some features require additional packages:

.. code-block:: bash

   # For LMDB data readers
   pip install lmdb

   # For image processing
   pip install Pillow

   # For plotting utilities
   pip install matplotlib

   # For colorized logging
   pip install colorama

Testing
-------

Run the test suite:

.. code-block:: bash

   python -m pytest tests/