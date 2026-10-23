import json
from pathlib import Path


history = json.loads(Path("dit_runs/lightweight_v4_50k_scratch_ddp/history.json").read_text())

print("== steps 295000-330000 (every 50) ==")
for entry in history:
    if 295000 <= entry["step"] <= 330000 and entry["step"] % 50 == 0:
        print(
            f"step {entry['step']:6d} loss {entry['loss']:.4f} "
            f"lr {entry['lr']:.3e} sps {entry['steps_per_second']:.1f}"
        )

first_bad = None
for entry in history:
    if entry["step"] > 100000 and entry["loss"] > 0.2:
        first_bad = entry
        break
print("\nfirst entry after 100k with loss > 0.2:", first_bad)
