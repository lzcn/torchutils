import codecs
import os.path

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


VERSION = get_version("torchutils/__init__.py")

setuptools.setup(
    name="torchutils",
    version=VERSION,
    author="Zhi Lu",
    author_email="zhilu@std.uestc.edu.cn",
    url="https://github.com/lzcn/torchutils",
    description="A bunch of personal utilities for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)

install_requires = [
    "attrs",
    "colorma",
    "numpy",
    "opencv-python",
    "pandas",
    "python-lmdb",
    "pyyaml",
    "scikit-learn",
]
