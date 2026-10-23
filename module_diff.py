from pathlib import Path

import torch

from lightweight_dit import DiTConfig, LightweightDiT


PATHS = {
    320000: Path("dit_runs/lightweight_v4_50k_scratch_ddp_continue/checkpoint-320000.pt"),
    330000: Path("dit_runs/lightweight_v4_50k_scratch_ddp_continue/checkpoint-330000.pt"),
    340000: Path("dit_runs/lightweight_v4_50k_scratch_ddp_continue/checkpoint-340000.pt"),
    400000: Path("dit_runs/lightweight_v4_50k_scratch_ddp_continue/checkpoint-400000.pt"),
}


def flat(model: LightweightDiT):
    out = {}
    for name, p in model.named_parameters():
        top = ".".join(name.split(".")[:3])
        out.setdefault(top, []).append(p.detach())
    return {k: torch.cat([x.flatten() for x in v]) for k, v in out.items()}


def main() -> None:
    config = DiTConfig(patch_size=2, hidden_size=384, depth=8, num_heads=8)
    states = {}
    for step, path in PATHS.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = LightweightDiT(config)
        model.load_state_dict(ck["model"])
        states[step] = flat(model)

    keys = sorted(states[320000])
    print(f"{'module':>42} {'L2@320k':>9} {'d330/320':>9} {'d340/320':>9} {'d400/320':>9}")
    for key in keys:
        ref = states[320000][key]
        ref_l2 = ref.pow(2).sum().sqrt().item()
        ds = []
        for step in (330000, 340000, 400000):
            delta = (states[step][key] - ref).pow(2).sum().sqrt().item()
            ds.append(delta / (ref_l2 + 1e-12))
        print(f"{key:>42} {ref_l2:9.3f} {ds[0]:9.4f} {ds[1]:9.4f} {ds[2]:9.4f}")


if __name__ == "__main__":
    main()
