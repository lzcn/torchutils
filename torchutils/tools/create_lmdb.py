import argparse
import logging
import os

import lmdb
from tqdm import tqdm

from torchutils.io.filesystem import scan_files

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def create_lmdb(dst: str, src: str, key: str = "relpath"):
    """Convert image files in a directory into an LMDB database."""
    LOGGER.info("Creating LMDB to %s", dst)

    suffix = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif")
    file_list = scan_files(src, suffix, recursive=True, relpath=False)

    if key == "relpath":
        key_list = [os.path.relpath(f, start=src) for f in file_list]
    elif key == "filename":
        key_list = [os.path.basename(f) for f in file_list]
    else:
        key_list = file_list

    env = lmdb.open(dst, map_size=2**40)

    with env.begin(write=True) as txn:
        for k, fn in tqdm(
            zip(key_list, file_list), total=len(file_list), desc="Writing LMDB"
        ):
            with open(fn, "rb") as f:
                img_data = f.read()
            txn.put(k.encode("utf-8"), img_data)

    env.close()
    LOGGER.info("Finished creating LMDB.")


def main():
    parser = argparse.ArgumentParser(description="Create an LMDB dataset from images.")
    parser.add_argument("--src", type=str, required=True, help="Source image directory")
    parser.add_argument("--dst", type=str, required=True, help="Destination LMDB path")
    parser.add_argument(
        "--key",
        type=str,
        default="relpath",
        choices=["relpath", "filename", "fullpath"],
        help="Key format in LMDB",
    )
    args = parser.parse_args()

    create_lmdb(dst=args.dst, src=args.src, key=args.key)


if __name__ == "__main__":
    main()
