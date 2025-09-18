import logging
import os
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


def check_exists(paths: Union[str, List[str]], mode: str = "any", verbose: bool = False) -> bool:
    """
    Check whether file(s) or folder(s) exist.

    Args:
        paths (Union[str, List[str]]): A single path or list of paths.
        mode (str): 'any' or 'all'. If 'all', all paths must exist.
        verbose (bool): If True, logs the existence status.

    Returns:
        bool: True if existence condition is met.

    Example:
        >>> check_exists(["./data", "./config.json"], mode="all")
        True
    """
    paths = [paths] if isinstance(paths, str) else paths
    flags = [os.path.exists(p) for p in paths]

    if verbose:
        for p, ok in zip(paths, flags):
            logger.info("%s: %s", p, "exists" if ok else "not found")

    return all(flags) if mode == "all" else any(flags)


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
