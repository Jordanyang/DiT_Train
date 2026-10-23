#!/usr/bin/env python3
"""Prepare the CIFAR-10 train split as 32x32 RGB PNGs (no labels).

Two sources are supported:

1. (default) Download CIFAR-10 via torchvision, then export the 50000-image
   train split to PNG.
2. Extract from an existing parquet file with an ``img.bytes`` column
   (--from-parquet datasets_cache/cifar10-train.parquet).

Usage:
    python prepare_data.py                          # download + export
    python prepare_data.py --out images_32x32_50000
    python prepare_data.py --from-parquet datasets_cache/cifar10-train.parquet
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path


def validate_and_write(raws: list[bytes], out: Path) -> None:
    from PIL import Image

    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to modify non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if len(raws) != 50000:
        raise RuntimeError(f"Expected 50000 images, got {len(raws)}")

    for count, raw in enumerate(raws, 1):
        with Image.open(BytesIO(raw)) as image:
            image.load()
            if image.size != (32, 32):
                raise RuntimeError((count, image.size))
            if image.mode != "RGB":
                raise RuntimeError((count, image.mode))
        (out / f"{count:05d}.png").write_bytes(raw)

    files = sorted(out.glob("*.png"))
    if len(files) != 50000:
        raise RuntimeError((len(files), out))
    print(f"Saved and validated {len(files)} images to {out.resolve()}")


def export_from_parquet(parquet_path: Path, out: Path) -> None:
    import fastparquet

    parquet_file = fastparquet.ParquetFile(str(parquet_path))
    # NOTE: read the FULL table. A common copy-paste bug is ``pf[:500]``,
    # which only reads the first 500 rows.
    frame = parquet_file.to_pandas(columns=["img.bytes"])
    validate_and_write(frame["img.bytes"].tolist(), out)


def export_from_torchvision(cache_dir: Path, out: Path) -> None:
    from torchvision.datasets import CIFAR10

    train = CIFAR10(root=str(cache_dir), train=True, download=True)
    raws = []
    for image, _label in train:  # labels are intentionally discarded
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        raws.append(buffer.getvalue())
    validate_and_write(raws, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("images_32x32_50000"))
    parser.add_argument("--from-parquet", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets_cache"))
    args = parser.parse_args()

    if args.from_parquet is not None:
        export_from_parquet(args.from_parquet, args.out)
    else:
        export_from_torchvision(args.cache_dir, args.out)


if __name__ == "__main__":
    main()
