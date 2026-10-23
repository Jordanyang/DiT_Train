from pathlib import Path
import math
import random

import torch
from PIL import Image


def load_tiles(path: Path, n: int = 64) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    grid = (
        torch.tensor(list(image.getdata()), dtype=torch.float32)
        .div_(255.0)
        .view(height, width, 3)
        .permute(2, 0, 1)[None]
    )
    nrow = math.isqrt(n)
    stride = 34  # 32 px tile + 2 px padding
    tiles = []
    for row in range(nrow):
        for col in range(nrow):
            tiles.append(grid[:, :, row * stride + 2 : (row + 1) * stride, col * stride + 2 : (col + 1) * stride])
    return torch.cat(tiles)


def metrics(x: torch.Tensor):
    x = x.float() * 2 - 1
    lap = (
        -4 * x[:, :, 1:-1, 1:-1]
        + x[:, :, :-2, 1:-1]
        + x[:, :, 2:, 1:-1]
        + x[:, :, 1:-1, :-2]
        + x[:, :, 1:-1, 2:]
    )
    y = x.roll(-1, dims=3)
    a = x - x.mean()
    b = y - y.mean()
    corr = torch.dot(a.flatten(), b.flatten()) / (a.flatten().norm() * b.flatten().norm() + 1e-12)
    return x.mean().item(), x.std().item(), corr.item(), lap.pow(2).mean().sqrt().item()


def main() -> None:
    sample_dir = Path("dit_runs/lightweight_v4_50k_scratch_ddp/samples")
    print(f"{'stage':>10} {'mean':>7} {'std':>7} {'adjCorr':>8} {'lapRMS':>7}")
    for path in sorted(sample_dir.glob("preview_step*_grid.png")):
        step = path.stem.split("step")[1].split("_")[0]
        m = metrics(load_tiles(path))
        print(f"{step:>10} {m[0]:7.4f} {m[1]:7.4f} {m[2]:8.4f} {m[3]:7.4f}")
    m = metrics(load_tiles(sample_dir / "final_step400000_grid.png"))
    print(f"{'final400k':>10} {m[0]:7.4f} {m[1]:7.4f} {m[2]:8.4f} {m[3]:7.4f}")

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
    print(f"{'real':>10} {m[0]:7.4f} {m[1]:7.4f} {m[2]:8.4f} {m[3]:7.4f}")


if __name__ == "__main__":
    main()
