torchutils.data
===============

.. currentmodule:: torchutils.data

This module defines the the data reader and some utility functions.

Data Readers
------------

The instance of :class:`~DataReader` class is callable. It calls the method :meth:`~DataReader.load` to load data.
The ``key`` argument has different meanings in different readers. The supported data readers are:

- :class:`~ImagePILReader`: A data reader with PIL backend for image. Data is read directly from path using PIL.
- :class:`~TensorLMDBReader`: A data reader with LMDB backend for :class:`numpy.ndarray`. Data is saved in ``key:value`` format.
- :class:`~ImageLMDBReader`: A data reader with LMDB backend for image raw data. Data is saved in ``key:value`` format.
- :class:`~TensorPKLReader`: A data reader with pickle backend for :class:`numpy.ndarray`. Data is saved in ``key:value`` format.

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

