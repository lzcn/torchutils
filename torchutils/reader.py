from abc import ABCMeta, abstractmethod
import logging
import os
import pickle
from typing import Any, Callable, Dict, Optional, Union

import lmdb
import numpy as np
import PIL.Image
import six
import torch
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

__all__ = [
    "getReader",
    "DataReader",
    "ImageLMDBReader",
    "ImagePILReader",
    "TensorLMDBReader",
    "TensorPKLReader",
]


def _load_pkl_data(path: str) -> Dict[str, np.ndarray]:
    """Load dictionary of name -> ndarray from pickle file."""
    with open(os.path.expanduser(path), "rb") as f:
        return pickle.load(f)


def _open_lmdb_env(path: str) -> lmdb.Environment:
    """Open LMDB environment for reading."""
    return lmdb.open(
        os.path.expanduser(path),
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def _get_transforms(data_transform: Union[str, Callable, None]) -> Callable:
    """
    Parse data transform configuration.

    Args:
        data_transform (Union[str, Callable, None]): A function or eval-able string of transforms.

    Returns:
        Callable: A transform function.
    """
    if isinstance(data_transform, str):
        data_transform = eval(data_transform)
        if isinstance(data_transform, Callable):
            return data_transform
        return transforms.Compose(data_transform)
    return data_transform if callable(data_transform) else lambda x: x


class DataReader(metaclass=ABCMeta):
    """
    Abstract base class for data readers.

    Args:
        path (str): Root path to data.
        data_transform (Callable, optional): Data preprocessing function.
        default (Any, optional): Fallback value if key not found.

    Usage:
        >>> reader = YourReader(path="/some/data")
        >>> output = reader("filename_or_key")
    """

    def __init__(
        self,
        path: str,
        data_transform: Optional[Callable] = None,
        default: Optional[Any] = None,
    ):
        self.path = path
        self.data_transform = data_transform
        self.default = default

    @abstractmethod
    def load(self, key: str) -> Any:
        """Load raw data by key."""
        raise NotImplementedError

    def __call__(self, key: str) -> Any:
        data = self.load(key)
        if data is None:
            return self.default
        return self.data_transform(data) if self.data_transform else data


class TensorLMDBReader(DataReader):
    """Reader for tensor data with LMDB backend."""

    def __init__(self, path, data_transform=None, default=None):
        super().__init__(path, data_transform=data_transform, default=default)
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
            if buf is None:
                return None
        feature = np.frombuffer(buf, dtype=np.float32).reshape(1, -1)
        return torch.from_numpy(feature.copy()).view(-1)


class ImagePILReader(DataReader):
    """Reader for image.

    Args:
        path (str): data root for images
        data_transform (Callable, optional): data transform. Defaults to lambda x: x.
    """

    def __init__(self, path, data_transform=None, default=None):
        super().__init__(path, data_transform=data_transform, default=default)

    def load(self, name: str) -> PIL.Image.Image:
        """Load PIL.Image

        Args:
            name (str): relative image path under self.path

        Returns:
            PIL.Image.Image: loaded image before transform
        """
        # read from raw image
        path = os.path.join(self.path, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            img = PIL.Image.open(f).convert("RGB")
        return img


class ImageLMDBReader(DataReader):
    """Reader for image with LMDB backend.

    Args:
        path (str): folder for LMDB data

    """

    def __init__(self, path, data_transform=None, default=None):
        super().__init__(path, data_transform=data_transform, default=default)
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
            if imgbuf is None:
                return None
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = PIL.Image.open(buf).convert("RGB")
        return img


class TensorPKLReader(DataReader):
    """Reader for tensor data."""

    def __init__(self, path, data_transform=None, default=None):
        super().__init__(path, data_transform=data_transform, default=default)
        self._data = _load_pkl_data(path)

    def load(self, name) -> torch.Tensor:
        if name not in self._data:
            return None
        feature = self._data[name].astype(np.float32)
        return torch.from_numpy(feature.copy())


class DummyReader(DataReader):
    """Dummy data reader.

    Args:
        path (str): data root for images
        data_transform (Callable, optional): data transform. Defaults to lambda x:x.
    """

    def __init__(self, path, data_transform=None, default=None):
        super().__init__(path, data_transform=data_transform, default=default)
        if default is None:
            self.default = torch.zeros(1)

    def load(self, name: str):
        return None


_READERS = {
    "ImageLMDB": ImageLMDBReader,
    "ImagePIL": ImagePILReader,
    "TensorLMDB": TensorLMDBReader,
    "TensorPKL": TensorPKLReader,
    "Dummy": DummyReader,
}


def getReader(reader: str, path: str, data_transform: Union[str, Callable, None] = None) -> DataReader:
    """
    Reader factory by type string.

    Args:
        reader (str): One of "ImageLMDB", "ImagePIL", "TensorLMDB", "TensorPKL", "Dummy".
        path (str): Path to dataset.
        data_transform (Union[str, Callable], optional): Preprocessing transform.

    Returns:
        DataReader: Instance of reader.

    Example:
        >>> from torchvision import transforms
        >>> reader = getReader(
        ...     reader="ImagePIL",
        ...     path="/data/images",
        ...     data_transform=transforms.ToTensor()
        ... )
        >>> img = reader("cat.jpg")
    """
    if reader not in _READERS:
        raise ValueError(f"reader must be one of: {', '.join(_READERS.keys())}")
    return _READERS[reader](path, _get_transforms(data_transform))
