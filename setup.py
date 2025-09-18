import codecs
import os

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    try:
        with codecs.open(os.path.join(here, rel_path), "r", encoding="utf-8") as fp:
            return fp.read()
    except FileNotFoundError:
        raise RuntimeError(f"File {rel_path} not found.")


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


VERSION = get_version("torchutils/__init__.py")

with open("README.md", encoding="utf-8") as fh:
    README = fh.read()

requirements = [
    "lmdb",
    "numpy",
    "pandas",
    "pillow",
    "matplotlib",
    "scikit-learn",
    "scipy",
    "tensorboard",
    "torch",
    "torchvision",
    "tqdm",
]

setup(
    # Metadata
    name="torchutils",
    version=VERSION,
    author="Zhi Lu",
    author_email="zhilu@std.uestc.edu.cn",
    url="https://github.com/lzcn/torchutils",
    description="A bunch of personal utilities for PyTorch",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.6",
    # Package info
    packages=find_packages(
        exclude=(
            "tests",
            "tests.*",
        )
    ),
    zip_safe=True,
    install_requires=requirements,
)
