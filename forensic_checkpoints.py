from pathlib import Path

import torch

from lightweight_dit import DDIM, DiTConfig, ImageFolderTensor, LightweightDiT


CHECKPOINTS = [
    ("run1", Path("dit_runs/lightweight_v4_50k_scratch_ddp/checkpoint-200000.pt")),
    ("run1", Path("dit_runs/lightweight_v4_50k_scratch_ddp/checkpoint-280000.pt")),
    ("run1", Path("dit_runs/lightweight_v4_50k_scratch_ddp/checkpoint-300000.pt")),
    ("run2", Path("dit_runs/lightweight_v4_50k_scratch_ddp_continue/checkpoint-310000.pt")),
    ("run2", Path("dit_runs/lightweight_v4_50k_scratch_ddp_continue/checkpoint-320000.pt")),
    ("run2", Path("dit_runs/lightweight_v4_50k_scratch_ddp_continue/checkpoint-330000.pt")),
    ("run2", Path("dit_runs/lightweight_v4_50k_scratch_ddp_continue/checkpoint-340000.pt")),
    ("run2", Path("dit_runs/lightweight_v4_50k_scratch_ddp_continue/checkpoint-400000.pt")),
]


def forward_stats(model: LightweightDiT, x0: torch.Tensor, diffusion: DDIM) -> dict:
    stats = {}
    with torch.no_grad():
        noise = torch.randn_like(x0)
        t = torch.randint(0, 400, (x0.shape[0],), device=x0.device)
        x_t = diffusion.add_noise(x0, noise, t)
        target = diffusion.velocity_target(x0, noise, t)
        for name, module, enabled in (
            ("bf16", model, True),
            ("fp32", model, False),
        ):
            if enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x_t, t).float()
            else:
                out = model(x_t, t).float()
            residual = out - target
            stats[name] = (
                out.abs().max().item(),
                out.std().item(),
                torch.isfinite(out).all().item(),
                residual.pow(2).mean().item(),
            )
    return stats


def main() -> None:
    device = torch.device("cuda:0")
    dataset = ImageFolderTensor(Path("images_32x32_50000"), 32)
    x0 = torch.stack([dataset[i] for i in range(0, 256, 2)]).to(device)
    diffusion = DDIM(400, device)
    config = DiTConfig(patch_size=2, hidden_size=384, depth=8, num_heads=8)

    print(
        f"{'ckpt':>8} {'step':>7} {'paramL2':>9} {'maxW':>8} "
        f"{'bf16|out|':>10} {'bf16std':>8} {'bf16mse':>9} "
        f"{'fp32|out|':>10} {'fp32std':>8} {'fp32mse':>9}"
    )
    for tag, path in CHECKPOINTS:
        ck = torch.load(path, map_location=device, weights_only=False)
        model = LightweightDiT(config).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        param_l2 = torch.sqrt(sum(p.pow(2).sum() for p in model.parameters())).item()
        max_w = max(p.abs().max().item() for p in model.parameters())
        stats = forward_stats(model, x0, diffusion)
        print(
            f"{tag:>8} {ck['step']:7d} {param_l2:9.2f} {max_w:8.3f} "
            f"{stats['bf16'][0]:10.2f} {stats['bf16'][1]:8.4f} {stats['bf16'][3]:9.4f} "
            f"{stats['fp32'][0]:10.2f} {stats['fp32'][1]:8.4f} {stats['fp32'][3]:9.4f}"
        )


if __name__ == "__main__":
    main()
