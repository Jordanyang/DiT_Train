import json
from pathlib import Path


history = json.loads(
    Path("dit_runs/lightweight_v4_50k_scratch_ddp_continue/history.json").read_text()
)
print("entries:", len(history), "range:", history[0]["step"], "-", history[-1]["step"])
print("max skipped_steps:", max(e.get("skipped_steps", 0) for e in history))

print("\n== every 1000 steps ==")
for entry in history:
    if entry["step"] % 1000 == 0:
        print(
            f"step {entry['step']:6d} loss {entry['loss']:.4f} "
            f"lr {entry['lr']:.3e} skipped {entry.get('skipped_steps', 0)}"
        )

first_bad = next((e for e in history if e["loss"] > 0.15), None)
print("\nfirst entry with loss > 0.15:", first_bad)
