import json
from pathlib import Path


history = json.loads(Path("dit_runs/lightweight_v4_50k_scratch_qknorm/history.json").read_text())
losses = [e["loss"] for e in history]
print("entries:", len(history), "range:", history[0]["step"], "-", history[-1]["step"])
print("max skipped_steps:", max(e.get("skipped_steps", 0) for e in history))
print("max loss:", max(losses), "at step", history[losses.index(max(losses))]["step"])
print("min loss:", min(losses), "at step", history[losses.index(min(losses))]["step"])
print("last 5:", [(e["step"], e["loss"]) for e in history[-5:]])

print("\n== every 20000 steps ==")
for entry in history:
    if entry["step"] % 20000 == 0:
        print(f"step {entry['step']:6d} loss {entry['loss']:.4f} lr {entry['lr']:.3e}")
