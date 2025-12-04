import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import yaml

logger = logging.getLogger(__name__)

_FORMAT_ALIASES = {
    "yml": "yaml",
}
_SUPPORTED_FORMATS = {"json", "yaml"}


def _normalize_format(fmt: Optional[str], file: Path) -> str:
    if fmt:
        fmt = fmt.lower()
    else:
        suffix = file.suffix.lstrip(".").lower()
        fmt = _FORMAT_ALIASES.get(suffix, suffix)
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{fmt}'. Expected one of: {sorted(_SUPPORTED_FORMATS)}"
        )
    return fmt


def load_config(file: Union[str, Path], fmt: Optional[str] = None) -> Any:
    """Load structured data from JSON or YAML files.

    Args:
        file: Path to the file.
        fmt: Optional format override ("json" or "yaml"). If omitted, inferred from suffix.

    Returns:
        Parsed Python object.
    """

    file = Path(file).expanduser()
    fmt = _normalize_format(fmt, file)
    with file.open("r") as f:
        if fmt == "json":
            return json.load(f)
        return yaml.safe_load(f)


def save_config(
    file: Union[str, Path],
    data: Any,
    fmt: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Save structured data to JSON or YAML files.

    Args:
        file: Target path.
        data: Serializable Python object.
        fmt: Optional format override ("json" or "yaml"). If omitted, inferred from suffix.
        overwrite: If False and file exists, skip with a warning.
    """

    file = Path(file).expanduser()
    fmt = _normalize_format(fmt, file)
    if file.exists() and not overwrite:
        logger.warning("%s already exists. Skipped.", file)
        return

    with file.open("w") as f:
        if fmt == "json":
            json.dump(data, f, indent=2)
        else:
            yaml.safe_dump(data, f, sort_keys=False)
