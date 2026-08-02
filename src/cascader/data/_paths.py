"""Where the collections live on disk.

Nothing large travels inside the wheel. The bundled WEAVE sample (50 substations,
under a megabyte) exists so the tests and the quick-start run offline; the full
collections are read from a directory the user points at.

Resolution order for the data directory:

1. the ``path`` argument of the loader,
2. the ``CASCADER_DATA_DIR`` environment variable,
3. ``./data`` relative to the working directory.

Expected layout under that directory::

    <data_dir>/
      train_weave_500.csv
      test_weave_500.csv
      fashion_mnist/          # the four idx-ubyte.gz files, downloaded on demand
      cifar10/                # cifar-10-batches-py, downloaded on demand
      celeba_features/        # X_train.npy y_train.npy X_test.npy y_test.npy
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

__all__ = ["data_dir", "bundled_dir", "require", "MissingDataError"]

_BUNDLED = Path(__file__).resolve().parent / "_bundled"


class MissingDataError(FileNotFoundError):
    """Raised when a collection is not bundled and not present on disk."""


def bundled_dir() -> Path:
    """Directory of the files shipped inside the wheel."""
    return _BUNDLED


def data_dir(path: Optional[str | Path] = None) -> Path:
    """Resolve the data directory. See the module docstring for the order."""
    if path is not None:
        return Path(path).expanduser().resolve()
    env = os.environ.get("CASCADER_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path("data").resolve()


def require(target: Path, what: str, how: str) -> Path:
    """Return ``target`` if it exists, otherwise explain how to obtain it."""
    if target.exists():
        return target
    raise MissingDataError(
        f"{what} not found at {target}.\n"
        f"{how}\n"
        "Then point CASCADER_DATA_DIR at the directory holding it, or pass "
        "path= to the loader."
    )
