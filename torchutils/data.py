import logging
import os
import pickle
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Union

import lmdb
import numpy as np
import PIL
import six
import torch
from torchvision import transforms
from tqdm import tqdm

from torchutils.files import scan_files
from torchutils.param import DataReaderParam

LOGGER = logging.getLogger(__name__)

__all__ = [
    "create_lmdb",
    "DataReader",
    "getReader",
    "ImageLMDBReader",
    "ImagePILReader",
    "ResizeToSquare",
    "TensorLMDBReader",
    "TensorPKLReader",
]


def create_lmdb(dst: str, src: Union[str, dict], key="relpath"):
    """Convert the image in src to dst in `LMDB`_ format

    .. _LMDB: https://lmdb.readthedocs.io/en/release/

    Args:
        dst (str): folder to save lmdb file.
        src (Union[str, dict]): folder or dict for (key, file) pairs
        key (str, optional): Defaults to "relpath". If src is a directory, it decides which key is used.

            - "relpath": use the relative path for key
            - "filename": use filename for key
            - "others": use fullpath for key

    Examples:

        .. code-block:: python

            dst = "/path/to/lmdb"
            src = "/path/to/images"
            create(dst, src, key="relpath")

    It reads images in ``src`` folder and creates two files under ``dst`` folder:
        - ``data.mdb``
        - ``lock.mdb``
    The key of each image is the relative path to ``src``.

    """
    LOGGER.info("Creating LMDB to %s", dst)
    if isinstance(src, dict):
        # assume all values are filenames
        file_list = src.values()
        key_list = src.keys()
    elif isinstance(src, str):
        # assume src is a directory
        suffix = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif")
        file_list = scan_files(src, suffix, recursive=True, relpath=False)
        if key == "relpath":
            key_list = [os.path.relpath(src, f) for f in file_list]
        elif key == "filename":
            key_list = [os.path.basename(f) for f in file_list]
        else:
            key_list = file_list
    else:
        raise ValueError("argument src must be a dict or directory path, but get {}".format(type(src)))
    env = lmdb.open(dst, map_size=2 ** 40)
    # open json file
    with env.begin(write=True) as txn:
        for k, fn in tqdm(zip(key_list, file_list), total=len(key_list)):
            with open(fn, "rb") as f:
                img_data = f.read()
                txn.put(k.encode("utf-8"), img_data)
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
        data_transform (Callable, optional): callable function for data transform. Defaults to lambdax:x
    """

    def __init__(self, path: str, data_transform: Callable = None):
        self.path = path
        self.data_transform = data_transform

    @abstractmethod
    def load(self, key) -> Any:
        """Load raw data.

        Args:
            key (str): The key is different for different readers.

        Returns:
            Any: raw data before data transform.
        """
        raise NotImplementedError

    def __call__(self, key):
        data = self.load(key)
        if self.data_transform is None:
            return data
        return self.data_transform(data)


class TensorLMDBReader(DataReader):
    """Reader for tensor data with LMDB backend."""

    def __init__(self, path, data_transform=None):
        super().__init__(path, data_transform=data_transform)
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
        return torch.from_numpy(feature.copy()).view(-1)


class ImagePILReader(DataReader):
    """Reader for image.

    Args:
        path (str): data root for images
        data_transform (Callable, optional): data transform. Defaults to lambdax:x.
    """

    def __init__(self, path, data_transform=None):
        super().__init__(path, data_transform=data_transform)

    def load(self, name: str) -> PIL.Image.Image:
        """Load PIL.Image

        Args:
            name (str): relative image path under self.path

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

    def __init__(self, path, data_transform=None):
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

    def __init__(self, path, data_transform=None):
        super().__init__(path, data_transform=data_transform)
        self._data = _load_pkl_data(path)

    def load(self, name) -> torch.Tensor:
        feature = self._data[name].astype(np.float32)
        return torch.from_numpy(feature.copy())


class DummyReader(DataReader):
    """Dummy data reader.

    Args:
        path (str): data root for images
        data_transform (Callable, optional): data transform. Defaults to lambdax:x.
    """

    def __init__(self, path, data_transform=None):
        super().__init__(path, data_transform=None)

    def load(self, name: str):
        data = torch.zeros(1)
        return data


_Readers = {
    "ImageLMDB": ImageLMDBReader,
    "ImagePIL": ImagePILReader,
    "TensorLMDB": TensorLMDBReader,
    "TensorPKL": TensorPKLReader,
    "Dummy": DummyReader,
}


def _get_transforms(data_transform):
    if isinstance(data_transform, str):
        data_transform = eval(data_transform)
        if isinstance(data_transform, Callable):
            return data_transform
        else:
            return transforms.Compose(data_transform)
    elif isinstance(data_transform, Callable):
        return data_transform
    else:
        return lambda x: x


def getReader(
    reader: str = None, path: str = None, data_transform: Union[str, Callable] = None, param: DataReaderParam = None
) -> DataReader:
    r"""Factory for DataReader.

    There are following types of data readers:
        - "ImageLMDB"(:class:`~torchutils.data.ImageLMDBReader`): image data with LMDB backend.
        - "ImagePIL"(:class:`~torchutils.data.ImagePILReader`): image data with PIL backend.
        - "TensorLMDB"(:class:`~torchutils.data.TensorLMDBReader`): tensor data with LMDB backend.
        - "TensorPKL"(:class:`~torchutils.data.TensorPKLReader`): tensor data saved with pickle.

    Args:
        reader (str): reader type
        path (str): data path
        data_transform (Union[str, Callable], optional): callable or string to eval. Defaults to None.
        param ([DataReaderParam], optional): use ReaderPraram to define. Defaults to None.

    Returns:
        DataReader: a callable instance of DataReader

    Examples:

        .. code-block:: python

            reader = "ImageLMDB"
            path = "/path/to/lmdb"
            data_transform = '''
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            '''
            reader = getReader(reader, path, data_transform)

        Within the above example, the ``data_trainsform`` is defined in string.

    """
    if param is not None:
        reader = param.reader
        path = param.path
        data_transform = param.data_transform
    if reader not in _Readers:
        raise ValueError("reader must be on of {}".format("|".join(_Readers.keys())))
    data_transform = _get_transforms(data_transform)
    return _Readers[reader](path, data_transform)
