"""Download the Fashion-MNIST IDX files (MIT license, Zalando Research) into data/."""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data"
    root.mkdir(exist_ok=True)
    for name in FILES:
        target = root / name
        if target.exists():
            print(f"exists  {name}")
            continue
        archive = root / f"{name}.gz"
        print(f"fetch   {name}.gz")
        urllib.request.urlretrieve(BASE + f"{name}.gz", archive)
        with gzip.open(archive, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        archive.unlink()


if __name__ == "__main__":
    main()
