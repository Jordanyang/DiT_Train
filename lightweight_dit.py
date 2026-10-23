#!/usr/bin/env python3
"""Train a small unconditional DiT on a directory of 32x32 RGB images."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image


@dataclass(frozen=True)
class DiTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    hidden_size: int = 256
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: int = 4
    time_dim: int = 256
    qk_norm: bool = False


class ImageFolderTensor(Dataset):
    def __init__(self, root: Path, image_size: int):
        self.paths = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    angles = t.float()[:, None] * frequencies[None, :]
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


class ModulatedTransformerBlock(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.num_heads = config.num_heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.proj = nn.Linear(width, width)
        hidden_mlp = width * config.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_mlp),
            nn.GELU(),
            nn.Linear(hidden_mlp, width),
        )
        self.ada_ln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, 6 * width),
        )
        if config.qk_norm:
            # RMSNorm on q/k keeps attention logits bounded and prevents the
            # slow q/k-norm growth -> softmax entropy collapse observed in the
            # 50k run after ~300k steps (ViT-22B / SD3 style QK-norm).
            head_dim = width // config.num_heads
            self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        nn.init.zeros_(self.ada_ln_mlp[-1].weight)
        nn.init.zeros_(self.ada_ln_mlp[-1].bias)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_ln_mlp(conditioning).chunk(6, dim=-1)
        )
        q, k, v = self.qkv(
            self.norm1(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        ).chunk(3, dim=-1)
        batch, tokens, width = q.shape
        heads = self.num_heads
        head_dim = width // heads
        q = q.view(batch, tokens, heads, head_dim).transpose(1, 2)
        k = k.view(batch, tokens, heads, head_dim).transpose(1, 2)
        v = v.view(batch, tokens, heads, head_dim).transpose(1, 2)
        if self.config.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        attention = F.scaled_dot_product_attention(q, k, v)
        attention = attention.transpose(1, 2).reshape(batch, tokens, width)
        x = x + gate_msa[:, None] * self.proj(attention)

        y = self.mlp(
            self.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        return x + gate_mlp[:, None] * y


class LightweightDiT(nn.Module):
    """An AdaLN-Zero diffusion transformer without classifier labels."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        width = config.hidden_size
        if width % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.patch_embed = nn.Conv2d(
            config.in_channels,
            width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        grid = config.image_size // config.patch_size
        self.num_tokens = grid * grid
        self.position = nn.Parameter(torch.zeros(1, self.num_tokens, width))
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList(
            ModulatedTransformerBlock(config) for _ in range(config.depth)
        )
        self.output_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.output_head = nn.Linear(width, config.in_channels * config.patch_size ** 2)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        self.position.data.normal_(mean=0.0, std=0.02)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        config = self.config
        patch = self.patch_embed(x_t)
        patch = patch.flatten(2).transpose(1, 2) + self.position
        conditioning = self.time_mlp(timestep_embedding(t, config.time_dim))
        for block in self.blocks:
            patch = block(patch, conditioning)
        patch = self.output_head(self.output_norm(patch))
        patch = patch.transpose(1, 2).view(
            x_t.shape[0],
            config.in_channels * config.patch_size * config.patch_size,
            config.image_size // config.patch_size,
            config.image_size // config.patch_size,
        )
        return F.pixel_shuffle(patch, config.patch_size)


class DDIM:
    def __init__(self, num_timesteps: int, device: torch.device):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device, dtype=torch.float64)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        ).float()

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.sqrt_alphas_cumprod[t][:, None, None, None] * x0
            + self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
        )

    def velocity_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        return (
            self.sqrt_alphas_cumprod[t][:, None, None, None] * noise
            - self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * x0
        )

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        steps: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        device = self.sqrt_alphas_cumprod.device
        x = torch.randn(shape, device=device, generator=generator)
        indices = torch.linspace(
            self.num_timesteps - 1, 0, steps, device=device
        ).round().long()

        for current_index, previous_index in zip(indices[:-1], indices[1:]):
            t = torch.full((shape[0],), int(current_index), device=device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model_output = model(x, t).float()
            alpha = self.alphas_cumprod[current_index].float()
            # At t_prev = 0, DDIM maps directly to clean data, so alpha_prev is 1.
            alpha_prev = (
                torch.tensor(1.0, device=device)
                if int(previous_index) == 0
                else self.alphas_cumprod[previous_index].float()
            )
            # v = sqrt(alpha) * epsilon - sqrt(1 - alpha) * x0
            x0_hat = torch.sqrt(alpha) * x - torch.sqrt(1 - alpha) * model_output
            x0_hat = x0_hat.clamp(-1, 1)
            epsilon_hat = (x - torch.sqrt(alpha) * x0_hat) / torch.sqrt(1 - alpha)
            x = (
                torch.sqrt(alpha_prev) * x0_hat
                + torch.sqrt(1 - alpha_prev) * epsilon_hat
            )

        return x.clamp(-1, 1)


def learning_rate(step: int, total_steps: int, peak_lr: float, warmup: int) -> float:
    if step < warmup:
        return peak_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 1e-5 + (peak_lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    for parameter, ema_parameter in zip(model.parameters(), ema_model.parameters()):
        ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1 - decay)


def save_samples(
    samples: torch.Tensor, directory: Path, prefix: str, individual_count: int
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    # Convert from model-space [-1, 1] to image-space [0, 1] explicitly.
    # torchvision's value_range is inactive unless normalize=True, so relying
    # on it silently clips negative pixels to zero and makes images very dark.
    images = (samples.float() + 1.0) * 0.5
    images = images.clamp(0.0, 1.0)
    grid = make_grid(images, nrow=math.isqrt(images.shape[0]), padding=2)
    grid_path = directory / f"{prefix}_grid.png"
    save_image(grid, grid_path)
    paths = [grid_path]
    for index, image in enumerate(images[:individual_count], start=1):
        path = directory / f"{prefix}_{index:02d}.png"
        save_image(image, path)
        paths.append(path)
    return paths


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def remove_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("images_32x32"))
    parser.add_argument("--output-dir", type=Path, default=Path("dit_runs/lightweight"))
    parser.add_argument("--train-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-timesteps", type=int, default=400)
    parser.add_argument("--prediction", choices=("epsilon", "x0", "v"), default="v")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument(
        "--qk-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply RMSNorm to attention q/k (bounded logits, no entropy collapse)",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--latest-checkpoint-interval", type=int, default=1000)
    parser.add_argument("--preview-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint path (defaults to <output-dir>/latest.pt)",
    )
    parser.add_argument(
        "--loss-spike-factor",
        type=float,
        default=6.0,
        help="Skip the optimizer step when loss exceeds this factor times the "
        "median of recent finite losses (<=0 disables the guard)",
    )
    parser.add_argument(
        "--guard-interval",
        type=int,
        default=10,
        help="Run the synchronous loss-spike guard every N steps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    config = DiTConfig(
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        qk_norm=args.qk_norm,
    )
    if config.image_size % config.patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")
    model = LightweightDiT(config).to(device)
    ema_model = LightweightDiT(config).to(device)
    ema_model.load_state_dict(model.state_dict(), strict=True)
    ema_model.requires_grad_(False)
    training_model = model
    if distributed:
        training_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    checkpoint_path = args.checkpoint if args.checkpoint is not None else args.output_dir / "latest.pt"
    start_step = 0

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        if rank == 0:
            print(f"Resumed checkpoint from step {start_step}", flush=True)

    # Give each rank its own RNG stream for diffusion noise / timesteps. Model
    # parameters stay identical because DDP broadcasts rank 0's initial state.
    torch.manual_seed(args.seed + rank)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "training.lock"
    if rank == 0:
        if not args.sample_only:
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(lock_fd, f"{os.getpid()}\n".encode())
                os.close(lock_fd)
            except FileExistsError as error:
                raise RuntimeError(
                    f"Another training process appears to be active: {lock_path}"
                ) from error
        with (args.output_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump({"model": asdict(config), "training": vars(args) | {
                "data_dir": str(args.data_dir),
                "output_dir": str(args.output_dir),
            }}, file, indent=2, default=str)
    if distributed:
        dist.barrier()

    if rank == 0:
        print(
            f"world_size={world_size} rank={rank} local_rank={local_rank} device={device}",
            flush=True,
        )
        print(
            f"Model: {count_parameters(model) / 1e6:.2f}M parameters; "
            f"tokens={model.num_tokens}; device={device}",
            flush=True,
        )

    diffusion = DDIM(args.num_timesteps, device)
    if not args.sample_only and start_step < args.train_steps:
        dataset = ImageFolderTensor(args.data_dir, config.image_size)
        if rank == 0:
            print(f"Dataset: {len(dataset)} images", flush=True)

        sampler = None
        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        batches = iter(loader)
        epoch = 0
        model.train()
        began = time.time()
        running_loss = 0.0
        running_count = 0
        recent_losses: list[float] = []
        skipped_steps = 0
        history = []
        history_path = args.output_dir / "history.json"
        if rank == 0 and history_path.exists() and start_step > 0:
            try:
                with history_path.open("r", encoding="utf-8") as file:
                    history = json.load(file)
            except (json.JSONDecodeError, OSError):
                history = []

        for step in range(start_step + 1, args.train_steps + 1):
            try:
                x0 = next(batches)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                batches = iter(loader)
                x0 = next(batches)
            x0 = x0.to(device, non_blocking=True)
            noise = torch.randn_like(x0)
            t = torch.randint(
                0,
                args.num_timesteps,
                (x0.shape[0],),
                device=device,
            )
            x_t = diffusion.add_noise(x0, noise, t)
            target = (
                noise
                if args.prediction == "epsilon"
                else diffusion.velocity_target(x0, noise, t)
                if args.prediction == "v"
                else x0
            )

            lr = learning_rate(
                step, args.train_steps, args.learning_rate, args.warmup_steps
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = training_model(x_t, t)
                loss = F.mse_loss(prediction.float(), target.float())

            loss.backward()

            # Synchronous loss-spike guard, run only every guard-interval steps
            # to avoid per-step host syncs / extra collectives that serialize
            # with DDP's gradient all-reduce. Uses the cross-rank mean loss so
            # every rank takes exactly the same branch (otherwise the DDP
            # collectives of later steps desynchronize and NCCL deadlocks).
            if args.loss_spike_factor > 0 and step % args.guard_interval == 0:
                guard_loss = loss.detach().clone()
                if distributed:
                    dist.all_reduce(guard_loss, op=dist.ReduceOp.SUM)
                    guard_loss.div_(world_size)
                guard_value = guard_loss.item()
                median_loss = statistics.median(recent_losses) if recent_losses else None
                if (
                    not math.isfinite(guard_value)
                    or median_loss is not None
                    and guard_value > args.loss_spike_factor * median_loss
                ):
                    optimizer.zero_grad(set_to_none=True)
                    skipped_steps += 1
                    if rank == 0:
                        print(
                            f"step {step:05d}: skipped bad loss {guard_value:.4f} "
                            f"(median={median_loss}, total skipped={skipped_steps})",
                            flush=True,
                        )
                    continue
                recent_losses.append(guard_value)
                if len(recent_losses) > 100:
                    recent_losses.pop(0)

            # Asynchronous per-step NaN/Inf sanitization: NaN grads become 0 and
            # clip_grad_norm_ then maps any remaining Inf-norm case to ~zero
            # updates, so a bad batch can never poison the weights or EMA. This
            # needs no host synchronization. In DDP, grads are already averaged
            # after backward, hence identical on all ranks.
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.nan_to_num_()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(model, ema_model, args.ema_decay)

            if rank == 0:
                running_loss += loss.item()
                running_count += 1
                if step % 50 == 0 or step == args.train_steps:
                    average = running_loss / running_count
                    elapsed = time.time() - began
                    sps = running_count / elapsed
                    print(
                        f"step {step:05d}/{args.train_steps} "
                        f"loss {average:.4f} lr {lr:.6f} "
                        f"{sps:.2f} steps/s",
                        flush=True,
                    )
                    history.append(
                        {
                            "step": step,
                            "loss": average,
                            "lr": lr,
                            "steps_per_second": sps,
                            "skipped_steps": skipped_steps,
                        }
                    )
                    running_loss = 0.0
                    running_count = 0
                    began = time.time()

            if (
                step % args.latest_checkpoint_interval == 0
                or step == args.train_steps
            ):
                if rank == 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config": asdict(config),
                    }
                    torch.save(checkpoint, checkpoint_path)
                    with history_path.open("w", encoding="utf-8") as file:
                        json.dump(history, file, indent=2)
                    if (
                        step % args.checkpoint_interval == 0
                        or step == args.train_steps
                    ):
                        # Keep immutable milestones in addition to latest.pt. This
                        # protects finished runs from being overwritten by stale jobs.
                        step_checkpoint = args.output_dir / f"checkpoint-{step:06d}.pt"
                        torch.save(checkpoint, step_checkpoint)
                if distributed:
                    dist.barrier()

                if (
                    args.preview_interval > 0
                    and (
                        step % args.preview_interval == 0
                        or step == args.train_steps
                    )
                ):
                    # Sampling only runs on rank 0; the barriers keep the other
                    # rank from racing ahead in the DDP collective stream.
                    if distributed:
                        dist.barrier()
                    if rank == 0:
                        preview_generator = torch.Generator(device=device)
                        preview_generator.manual_seed(args.seed + step)
                        model.eval()
                        preview = diffusion.sample(
                            ema_model,
                            (
                                args.num_samples,
                                config.in_channels,
                                config.image_size,
                                config.image_size,
                            ),
                            args.sample_steps,
                            preview_generator,
                        )
                        save_samples(
                            preview,
                            args.output_dir / "samples",
                            f"preview_step{step}",
                            0,
                        )
                        model.train()
                    if distributed:
                        dist.barrier()

    # Final sampling: rank 0 generates images while other ranks wait.
    if distributed:
        dist.barrier()
    if rank == 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        ema_model.eval()
        samples = diffusion.sample(
            ema_model,
            (args.num_samples, config.in_channels, config.image_size, config.image_size),
            args.sample_steps,
            generator,
        )
        paths = save_samples(
            samples,
            args.output_dir / "samples",
            f"final_step{args.train_steps}",
            min(16, args.num_samples),
        )
        print(f"Saved {len(paths)} output files under {args.output_dir / 'samples'}", flush=True)
        if lock_path.exists():
            remove_path(lock_path)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
