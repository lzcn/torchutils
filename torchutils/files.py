import logging
import os
from typing import List, Tuple, Union

import cv2
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _check_dir(folder, action="check", verbose=False):
    exists = os.path.isdir(folder)
    if not exists:
        if action == "mkdir":
            # make directories recursively
            os.makedirs(folder)
            exists = True
            LOGGER.info("folder '%s' has been created.", folder)
        if action == "check" and verbose:
            LOGGER.info("folder '%s' does not exist.", folder)
    else:
        if verbose:
            LOGGER.info("folder '%s' exist.", folder)
    return exists


def check_dirs(folders: str, action: str = "check", mode: str = "all", verbose: bool = False) -> bool:
    flags = []
    if action.lower() not in ["check", "mkdir"]:
        raise ValueError("{} not in ['check', 'mkdir']".format(action.lower()))
    if mode.lower() not in ["all", "any"]:
        raise ValueError("{} not in ['all', 'any']".format(mode.lower()))
    ops = {"any": any, "all": all}
    if verbose:
        LOGGER.info("Checked folder(s):")
    if isinstance(folders, str):
        flags.append(_check_dir(folders, action, verbose))
    else:
        for folder in folders:
            flags.append(_check_dir(folder, action, verbose))

    return ops[mode](flags)


def check_files(file_list, mode="any", verbose=False):
    """Check whether files exist, optional modes are ['all','any']."""
    n_file = len(file_list)
    opt_modes = ["all", "any"]
    ops = {"any": any, "all": all}
    if mode not in opt_modes:
        LOGGER.info("Wrong choice of mode, optional modes %s", opt_modes)
        return False
    exists = [os.path.isfile(fn) for fn in file_list]
    if verbose:
        LOGGER.info("names\t status")
        info = [file_list[i] + "\t" + str(exists[i]) for i in range(n_file)]
        LOGGER.info("\n".join(info))
    return ops[mode](exists)


def check_exists(lists, mode="any", verbose=False):
    """Check whether file(s)/folder(s) exist(s)."""
    n_file = len(lists)
    opt_modes = ["all", "any"]
    ops = {"any": any, "all": all}
    if mode not in opt_modes:
        LOGGER.info("Wrong choice of mode, optional modes %s", opt_modes)
        return False
    exists = [os.path.exists(fn) for fn in lists]
    if verbose:
        LOGGER.info("filename\t status")
        info = [lists[i] + "\t" + str(exists[i]) for i in range(n_file)]
        LOGGER.info("\n".join(info))
    return ops[mode](exists)


def scan_files(
    path: str = "./", suffix: Union[str, Tuple[str]] = "", recursive: bool = False, relpath: bool = False
) -> List:
    """Scan files under path which follows the PEP 471.

    Args:
        path (str, optional): target path. Defaults to "./".
        suffix (Union[str, Tuple[str]], optional): folder that ends with given suffix, it can also be a tuple. Defaults to "".
        recursive (bool, optional): scan files recursively. Defaults to False.
        relpath (bool, optional): return relative path. Defaults to False.

    Returns:
        List: list of files

    """

    def scantree(path):
        for entry in os.scandir(path):
            if not entry.name.startswith("."):
                if entry.is_dir(follow_symlinks=False):
                    yield from scantree(entry.path)
                else:
                    yield entry

    def scandir(path):
        for entry in os.scandir(path):
            if not entry.name.startswith(".") and entry.is_file():
                yield entry

    files = []
    scan = scantree if recursive else scandir
    for entry in scan(path):
        if entry.name.endswith(suffix):
            files.append(os.path.relpath(entry.path, path) if relpath else entry.path)
    return files


def scan_folders(
    path: str = "./", suffix: Union[str, Tuple[str]] = "", recursive: bool = False, relpath: bool = False
) -> List:
    """Scan folders under path which follows the PEP 471.

    Args:
        path (str, optional): target path. Defaults to "./".
        suffix (Union[str, Tuple[str]], optional): folder that ends with given suffix, it can also be a tuple. Defaults to "".
        recursive (bool, optional): scan files recursively. Defaults to False.
        relpath (bool, optional): return relative path. Defaults to False.

    Returns:
        List: list of folders

    """

    def scantree(path):
        for entry in os.scandir(path):
            if not entry.name.startswith("."):
                if entry.is_dir(follow_symlinks=False):
                    yield from scantree(entry.path)
            if entry.is_dir():
                yield entry

    def scandir(path):
        for entry in os.scandir(path):
            if not entry.name.startswith(".") and entry.is_dir():
                yield entry

    folders = []
    scan = scantree if recursive else scandir
    for entry in scan(path):
        if entry.name.endswith(suffix):
            folders.append(os.path.relpath(entry.path, path) if relpath else entry.path)
    return folders


def list_files(folder="./", suffix="", recursive=False):
    """Deprecated, use scan_files instead.

    Parameters
    ----------
    suffix: filename must end with suffix if given, it can also be a tuple
    recursive: if recursive, return sub-paths
    """
    files = []
    if recursive:
        for path, _, fls in os.walk(folder):
            files += [os.path.join(path, f) for f in fls if f.endswith(suffix)]
    else:
        files = [f for f in os.listdir(folder) if f.endswith(suffix)]
    return files


def read_csv(fn) -> np.array:
    return np.array(pd.read_csv(fn, dtype=np.int))


def save_csv(data, fn, cols):
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(fn, index=False)


def resize_image(src, dest, sizes=[224, 224]):
    """Resize fashion image.

    Resize rule: dilate the original image to square with all 0 filled.
    Cause all fashion image are with white background
    Then resize the image to given sizes [new_height, new_width]
    """
    img = cv2.imread(src)
    height, width, depth = img.shape
    ratio = 1.0 * height / width
    new_height = sizes[0]
    new_width = sizes[1]
    new_ratio = 1.0 * new_height / new_width
    if ratio > new_ratio:
        h = height
        w = int(height / new_ratio)
        new_image = np.zeros((h, w, depth)).astype(np.uint8)
        new_image[:] = 255
        new_p = int((w - width) / 2)
        new_image[:, new_p : new_p + width, :] = img
    else:
        h = int(new_ratio * height)
        w = width
        new_image = np.zeros((h, w, depth)).astype(np.uint8)
        new_image[:] = 255
        new_p = int((h - height) / 2)
        new_image[new_p : new_p + height, :, :] = img
    resized_img = cv2.resize(new_image, (new_width, new_height))
    cv2.imwrite(dest, resized_img)
