torchutils.data
===============

.. currentmodule:: torchutils.data

Module :mod:`torchutils.data` defines the data reader class and some utility functions.

Data Readers
------------

The instance of :class:`~DataReader` class is callable. It calls the method :meth:`~DataReader.load` to load data.
The ``key`` argument has different meanings in different readers.
Suppose data is saved in ``key:value`` manner. Specially, for data saved in file, the key is the relative path.

The supported data readers are:

- :class:`~ImageLMDBReader`: A data reader with LMDB backend for image raw data.
- :class:`~ImagePILReader`: A data reader with PIL backend for image.
- :class:`~TensorLMDBReader`: A data reader with LMDB backend for :class:`numpy.ndarray`.
- :class:`~TensorPKLReader`: A data reader with pickle backend for :class:`numpy.ndarray`.

Converting Images to LMDB
-------------------------

Currently, only one simple method, i.e. :meth:`create_lmdb` is provided for converting the images.
This method creates the LMDB data for images in one folder. And use the path of images as the key for retrieval.

Converting Images to Square
---------------------------

A new transform :class:`ResizeToSquare` for converting images to square in added.
The longest side (width or hight) is firstly resized to match the given size. The raito of image is fixed.
Then white pixes are padded to get a square image.

Module Reference
----------------

.. automodule:: torchutils.data
    :members:

