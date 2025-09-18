import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

logger = logging.getLogger(__name__)


def load_json(file: Union[str, Path]) -> Any:
    """Load JSON file.

    Args:
        file: JSON file path

    Returns:
        Parsed JSON object

    Example:
        >>> config = load_json("config.json")
        >>> print(config["name"])
    """
    file = Path(file).expanduser()
    with file.open("r") as f:
        return json.load(f)


def save_json(file: Union[str, Path], data: Any, overwrite: bool = False) -> None:
    """Save data to a JSON file.

    Args:
        file: File path
        data: Data to be serialized
        overwrite: If False and file exists, will skip saving

    Example:
        >>> data = {"name": "model", "version": 1}
        >>> save_json("output.json", data, overwrite=True)
    """
    file = Path(file).expanduser()
    if file.exists() and not overwrite:
        logger.warning("%s already exists. Skipped.", file)
        return
    with file.open("w") as f:
        json.dump(data, f)


def load_csv(
    file: Union[str, Path],
    skip_rows: int = 0,
    converter: Optional[Callable[[str], Any]] = None,
) -> List[List[Any]]:
    """Load CSV file.

    Args:
        file: File path
        skip_rows: Rows to skip from top
        converter: Optional callable to convert each element

    Returns:
        Parsed list of rows

    Example:
        >>> rows = load_csv("data.csv")
        >>> rows[0]
        ['id', 'score']

        >>> rows = load_csv("data.csv", converter=int)
        >>> rows[1]
        [1, 95]
    """
    file = Path(file).expanduser()
    with file.open("r", newline="") as f:
        reader = csv.reader(f)
        for _ in range(skip_rows):
            next(reader)
        data = list(reader)
        if converter:
            data = [list(map(converter, row)) for row in data]
    return data


def save_csv(
    file: Union[str, Path],
    data: List[List[Any]],
    header: Optional[List[str]] = None,
    overwrite: bool = False,
) -> None:
    """Save data to CSV file.

    Args:
        file: File path
        data: List of rows
        header: Optional list of column names
        overwrite: If False and file exists, will skip saving

    Example:
        >>> rows = [[1, 90], [2, 85]]
        >>> save_csv("scores.csv", rows, header=["id", "score"], overwrite=True)
    """
    file = Path(file).expanduser()
    if file.exists() and not overwrite:
        logger.warning("%s already exists. Skipped.", file)
        return

    with file.open("w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(data)
