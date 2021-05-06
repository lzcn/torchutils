import csv
import json
import numpy as np


def load_json(fn):
    """Load json data from file

    Args:
        fn (str): file name

    Returns:
        Any: data
    """
    with open(fn, "r") as f:
        data = json.load(f)
    return data


def save_json(fn, data):
    """Save data in json format.

    Args:
        fn (str): file name
        data (Any): data to save
    """
    with open(fn, "w") as f:
        json.dump(data, f)


def load_csv(fn, num_skip=0, converter=None):
    """Load data in csv format.

    Args:
        fn (str): file name
        num_skip (int, optional): number of lines to skip. Defaults to 0.
        converter (type, optional): convert str to desired type. Defaults to None.

    Returns:
        List: data
    """
    with open(fn, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for _ in range(num_skip):
            next(reader)
        data = list(reader)
        if converter is not None:
            data = [list(map(converter, line)) for line in data]
    return data


def save_csv(fn, data):
    """Save data in csv format.

    Args:
        fn (str): file name
        data (Any): data to save
    """
    with open(fn, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)
