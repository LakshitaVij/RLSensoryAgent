"""INTERNAL: export + build features for the 5 held-out subjects. Called only by src/final_eval.py.

Writes to data/heldout/clean, data/heldout/features and data/heldout/eval_only (separate from the development
folders). Refuses to run without the exact confirmation text.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
CONFIRM = "OPEN-HELD-OUT-ONCE"

if __name__ == "__main__":
    if "--confirm" not in sys.argv or sys.argv[sys.argv.index("--confirm") + 1] != CONFIRM:
        sys.exit("refusing: this opens the held-out subjects. It is only called by src/final_eval.py.")
    import json
    import build_features
    import export_subject
    held = json.loads((ROOT / "splits.json").read_text())["held_out"]
    H = ROOT / "data" / "heldout"
    for sid in held:
        export_subject.export(sid, allow_heldout=True, out_root=H / "clean")
        build_features.build(sid, allow_heldout=True, clean_root=H / "clean", out_root=H / "features", eval_root=H / "eval_only")
