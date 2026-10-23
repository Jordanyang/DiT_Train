from pathlib import Path

import torch

from lightweight_dit import DDIM, DiTConfig, ImageFolderTensor, LightweightDiT


CHECKPOINTS = [
    Path("dit_runs/lightweight_v4_50k_scratch_qknorm/checkpoint-100000.pt"),
    Path("dit_runs/lightweight_v4_50k_scratch_qknorm/checkpoint-300000.pt"),
    Path("dit_runs/lightweight_v4_50k_scratch_qknorm/checkpoint-400000.pt"),
]


def main() -> None:
    device = torch.device("cuda:0")
    dataset = ImageFolderTensor(Path("images_32x32_50000"), 32)
    x0 = torch.stack([dataset[i] for i in range(0, 256, 2)]).to(device)
    diffusion = DDIM(400, device)
    config = DiTConfig(patch_size=2, hidden_size=384, depth=8, num_heads=8, qk_norm=True)

    torch.manual_seed(0)
    noise = torch.randn_like(x0)
    t = torch.randint(0, 400, (x0.shape[0],), device=device)
    x_t = diffusion.add_noise(x0, noise, t)

    stats_all = {}
    for path in CHECKPOINTS:
        ck = torch.load(path, map_location=device, weights_only=False)
        model = LightweightDiT(config).to(device)
        model.load_state_dict(ck["model"])
        model.eval()

        block_stats = []
        hooks = []

        def make_hook(block, store):
            def hook(module, inputs, output):
                q, k, v = output.chunk(3, dim=-1)
                b, n, width = q.shape
                heads = config.num_heads
                head_dim = width // heads
                q = q.view(b, n, heads, head_dim).transpose(1, 2)
                k = k.view(b, n, heads, head_dim).transpose(1, 2)
                qn_raw = q.norm(dim=-1).mean().item()
                kn_raw = k.norm(dim=-1).mean().item()
                q = block.q_norm(q)
                k = block.k_norm(k)
                logits = (q @ k.transpose(-1, -2)) / (head_dim ** 0.5)
                probs = logits.softmax(dim=-1)
                entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean().item()
                store.append((logits.abs().max().item(), qn_raw, kn_raw, entropy))
            return hook

        for block in model.blocks:
            hooks.append(block.qkv.register_forward_hook(make_hook(block, block_stats)))

        with torch.no_grad():
            model(x_t, t)
        for h in hooks:
            h.remove()
        stats_all[ck["step"]] = block_stats
        del model

    for step in sorted(stats_all):
        print(f"\n== step {step} (post-norm logits actually used by attention) ==")
        for i, (lmax, qn, kn, ent) in enumerate(stats_all[step]):
            print(
                f"block{i}: logit_max {lmax:7.2f}  q_norm_raw {qn:6.2f}  "
                f"k_norm_raw {kn:6.2f}  entropy {ent:5.3f}"
            )


if __name__ == "__main__":
    main()
