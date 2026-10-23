import math
import random
import sys
from pathlib import Path

import torch
from PIL import Image

from analyze_samples import load_tiles, metrics


def main() -> None:
    print(f"{'source':>34} {'mean':>7} {'std':>7} {'adjCorr':>8} {'lapRMS':>7}")
    for path in sys.argv[1:]:
        m = metrics(load_tiles(Path(path)))
        print(f"{str(path):>34} {m[0]:7.4f} {m[1]:7.4f} {m[2]:8.4f} {m[3]:7.4f}")
    random.seed(0)
    paths = random.sample(sorted(Path("images_32x32_50000").glob("*.png")), 64)
    real = torch.stack(
        [
            torch.tensor(list(Image.open(p).convert("RGB").getdata()), dtype=torch.float32)
            .div_(255.0)
            .view(32, 32, 3)
            .permute(2, 0, 1)
            for p in paths
        ]
    )
    m = metrics(real)
    print(f"{'real (64 random)':>34} {m[0]:7.4f} {m[1]:7.4f} {m[2]:8.4f} {m[3]:7.4f}")


if __name__ == "__main__":
    main()
