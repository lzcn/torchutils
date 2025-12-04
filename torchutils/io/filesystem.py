import logging
import os
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


def check_exists(
    paths: Union[str, List[str]], mode: str = "any", verbose: bool = False
) -> bool:
    """Check whether file(s) or folder(s) exist.

    Args:
        paths: A single path or list of paths.
        mode: "any" or "all". If "all", all paths must exist.
        verbose: If True, logs the existence status.

    Returns:
        True if existence condition is met.
    """
    paths = [paths] if isinstance(paths, str) else paths
    flags = [os.path.exists(p) for p in paths]

    if verbose:
        for p, ok in zip(paths, flags):
            logger.info("%s: %s", p, "exists" if ok else "not found")

    return all(flags) if mode == "all" else any(flags)


def scan_files(
    path: str = "./",
    suffix: Union[str, Tuple[str]] = "",
    recursive: bool = False,
    relpath: bool = False,
) -> List[str]:
    """Scan files under path.

    Args:
        path: Target directory path.
        suffix: Filter files by suffix (can be a tuple of suffixes).
        recursive: If True, scan recursively.
        relpath: If True, return relative paths.

    Returns:
        List of file paths.
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
    path: str = "./",
    suffix: Union[str, Tuple[str]] = "",
    recursive: bool = False,
    relpath: bool = False,
) -> List[str]:
    """Scan folders under path.

    Args:
        path: Target directory path.
        suffix: Filter folders by suffix (can be a tuple of suffixes).
        recursive: If True, scan recursively.
        relpath: If True, return relative paths.

    Returns:
        List of folder paths.
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
