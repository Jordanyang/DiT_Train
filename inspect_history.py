import json
from pathlib import Path


history = json.loads(Path("dit_runs/lightweight_v4_50k_scratch_ddp/history.json").read_text())
print("total entries:", len(history), "first:", history[0]["step"], "last:", history[-1]["step"])

print("\n== every 2000 steps ==")
for entry in history:
    if entry["step"] % 2000 == 0:
        print(
            f"step {entry['step']:6d} loss {entry['loss']:.4f} "
            f"lr {entry['lr']:.3e} sps {entry['steps_per_second']:.1f}"
        )

print("\n== largest 50-step loss jumps ==")
jumps = []
for prev, curr in zip(history, history[1:]):
    jumps.append((curr["loss"] - prev["loss"], prev["step"], prev["loss"], curr["step"], curr["loss"]))
jumps.sort(reverse=True)
for delta, s0, l0, s1, l1 in jumps[:30]:
    print(f"{s0} ({l0:.4f}) -> {s1} ({l1:.4f})  delta +{delta:.4f}")

print("\n== loss > 0.2 after step 50000 ==")
for entry in history:
    if entry["step"] > 50000 and entry["loss"] > 0.2:
        print(f"step {entry['step']:6d} loss {entry['loss']:.4f} lr {entry['lr']:.3e}")
