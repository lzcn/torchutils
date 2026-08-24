import os
from pathlib import Path

__all__ = ["scan_files"]


def scan_files(
    path: str | Path = "./",
    suffix: str | tuple = (),
    recursive: bool = False,
    relpath: bool = False,
) -> list[str]:
    """Scan files under a path, skipping hidden entries (e.g. .DS_Store).

    Args:
        path: Target directory.
        suffix: Suffix filter, e.g. ".jpg" or (".jpg", ".png"). No filter if empty.
        recursive: Scan subdirectories if True.
        relpath: Return paths relative to ``path`` if True.

    Returns:
        List of file paths.
    """

    def iter_files(directory):
        for entry in os.scandir(directory):
            if entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=False):
                if recursive:
                    yield from iter_files(entry.path)
            else:
                yield entry

    return [
        os.path.relpath(e.path, path) if relpath else e.path
        for e in iter_files(path)
        if e.name.endswith(suffix)
    ]
