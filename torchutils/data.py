import argparse
import logging
import os
import pickle
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Optional, Union

import lmdb
import numpy as np
import PIL
import six
import torch
import torchvision.transforms as ts
from tqdm import tqdm

from . import files

LOGGER = logging.getLogger(__name__)

__all__ = [
    "getReader",
    "DataReader",
    "ImagePILReader",
    "TensorLMDBReader",
    "ImageLMDBReader",
    "TensorPKLReader",
    "create_lmdb",
    "ResizeToSquare",
]


def create_lmdb(dst, src):
    """Convert the image in src to dst in `LMDB`_ format

    .. _LMDB: https://lmdb.readthedocs.io/en/release/

    Args:
        dst (str): dst folder
        src (str): src folder

    Examples:

        .. code-block:: python

            dst = "/path/to/lmdb"
            src = "/path/to/images"
            create(dst, src)

        It convert the image in ``src`` folder to ``dst``, it will create two files:
        data.mdb, lock.mdb under ``dst`` folder.

    """
    LOGGER.info("Creating LMDB to %s", dst)
    suffix = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]
    image_list = files.list_files(src, suffix)
    env = lmdb.open(dst, map_size=2 ** 40)
    # open json file
    with env.begin(write=True) as txn:
        for image_name in tqdm(image_list):
            fn = os.path.join(src, image_name)
            with open(fn, "rb") as f:
                img_data = f.read()
                txn.put(image_name.encode("ascii"), img_data)
    env.close()
    LOGGER.info("Converted dst to LDMB")


class ResizeToSquare(object):
    """Resize an image to square with white background padded.

    It firstly resize the image with fixed ratio maing the longest side (w or h) meets
    given size. Then white pixels are padded.

    Args:
        size (int): output image size

    Example:

        .. code-block:: python

            trans = ResizeToSquare(291)
            im_resized = trans(im)

    """

    def __init__(self, size):
        self.s = size
        self.size = (size, size)

    def __call__(self, im):
        if im.size == self.size:
            return im
        w, h = im.size
        ratio = self.s / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        im = im.resize((new_w, new_h), PIL.Image.BILINEAR)
        new_im = PIL.Image.new("RGB", self.size, (255, 255, 255))
        new_im.paste(im, ((self.s - new_w) // 2, (self.s - new_h) // 2))
        return new_im

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.s)


_normalize_transform = ts.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_preset_transforms = {
    "augment": ts.Compose([ts.RandomHorizontalFlip(), ts.ToTensor(), _normalize_transform]),
    "totensor": ts.Compose([ts.ToTensor(), _normalize_transform]),
    "identity": lambda x: x,
}


def _get_transforms(transforms):
    if isinstance(transforms, str) and transforms in _preset_transforms:
        return _preset_transforms[transforms]
    elif isinstance(transforms, list):
        trans = []
        for x in transforms:
            if isinstance(x, str):
                trans.append(getattr(ts, x)())
            else:
                name, kwargs = x
                trans.append(getattr(ts, name)(**kwargs))
        return ts.Compose(trans)
    elif isinstance(transforms, Callable):
        return transforms
    else:
        raise KeyError("Transforms: '{}' is not defined".format(transforms))


def _load_pkl_data(path):
    fn = os.path.join(path)
    with open(fn, "rb") as f:
        data = pickle.load(f)
    return data


def _open_lmdb_env(path):
    return lmdb.open(path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)


class DataReader(metaclass=ABCMeta):
    """Callable reader for different types of data.

    Args:
        path (str): data path, different data reader requires different path
        data_transform (Callable, optional): callable function for data transform. Defaults to lambdax:x.
    """

    def __init__(self, path: str, data_transform: Callable = lambda x: x):
        self.path = path
        self.data_transform = data_transform

    @abstractmethod
    def load(self, key) -> Any:
        """Load raw data.

        Args:
            key (str): The key is different for different readers.

        Returns:
            Any: raw data before data transform,
        """
        pass

    def __call__(self, key):
        data = self.load(key)
        return self.data_transform(data)


class TensorLMDBReader(DataReader):
    """Reader for tensor data with LMDB backend."""

    def __init__(self, path, data_transform=None):
        super().__init__(path, data_transform=lambda x: x)
        self._env = _open_lmdb_env(path)

    def load(self, key) -> torch.Tensor:
        """Tensor reader with LMDB backend. Data saved in key-value pairs

        Args:
            key (str): key for data

        Returns:
            torch.Tensor: output tensor
        """
        with self._env.begin(write=False) as txn:
            buf = txn.get(key.encode())
        feature = np.frombuffer(buf, dtype=np.float32).reshape(1, -1)
        return torch.from_numpy(feature).view(-1)


class ImagePILReader(DataReader):
    """Reader for image.

    Args:
        path (str): data root for images
        data_transform (Callable, optional): data transform. Defaults to lambdax:x.
    """

    def __init__(self, path, data_transform=lambda x: x):
        super().__init__(path, data_transform=data_transform)

    def load(self, name: str) -> PIL.Image.Image:
        """Load PIL.Image

        Args:
            name (str): image path under data root

        Returns:
            PIL.Image.Image: loaded image before transform
        """
        # read from raw image
        path = os.path.join(self.path, name)
        with open(path, "rb") as f:
            img = PIL.Image.open(f).convert("RGB")
        return img


class ImageLMDBReader(DataReader):
    """Reader for image with LMDB backend.

    Args:
        path (str): folder for LMDB data

    """

    def __init__(self, path, data_transform=lambda x: x):
        super().__init__(path, data_transform=data_transform)
        self._env = _open_lmdb_env(path)

    def load(self, name: str) -> PIL.Image.Image:
        """Load an image from LMDB format, data saved in key-value pairs

        Args:
            name (str): key for the data

        Returns:
            PIL.Image.Image: loaded image before transform
        """
        with self._env.begin(write=False) as txn:
            imgbuf = txn.get(name.encode())
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = PIL.Image.open(buf).convert("RGB")
        return img


class TensorPKLReader(DataReader):
    """Reader for tensor data."""

    def __init__(self, path, data_transform=lambda x: x):
        super().__init__(path, data_transform=lambda x: x)
        self._data = self._load_pkl_data(path)

    def load(self, name):
        return torch.from_numpy(self._data[name].astype(np.float32))


def getReader(reader, path: str, data_transform: Optional[Union[str, list]] = "identity") -> DataReader:
    """Factory for DataReader.

    There are following types of data readers:
        - "TensorLMDB"(:class:`~torchutils.data.TensorLMDBReader`): tensor data with LMDB backend.
        - "ImageLMDB"(:class:`~torchutils.data.ImageLMDBReader`): image data with LMDB backend.
        - "TensorPKL"(:class:`~torchutils.data.TensorPKLReader`): tensor data saved with pickle.
        - "ImagePIL"(:class:`~torchutils.data.ImagePILReader`): image data with PIL backend.

    Args:
        reader (str): reader type
        path (str): data path
        data_transform (Optional[Union[str, list]], optional): data transform. Defaults to "identity".

    Returns:
        DataReader: a callable instance of DataReader

    TODO:
        Add detailed ducomentation for ``data_transform``

    """
    _Readers = {
        "TensorLMDB": TensorLMDBReader,
        "ImageLMDB": ImageLMDBReader,
        "TensorPKL": TensorPKLReader,
        "ImagePIL": ImagePILReader,
    }
    data_transform = _get_transforms(data_transform)
    return _Readers[reader](path, data_transform)
