"""Pick the 2 validation subjects (of the 9 usable train subjects) once: seed 42, integer-sorted ids.

Refuses to overwrite val_split.json. Opens no data files.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "val_split.json"
if OUT.exists():
    sys.exit(f"{OUT} already exists; the validation split is frozen. Not overwriting.")

split = json.loads((ROOT / "splits.json").read_text())
excluded = json.loads((ROOT / "excluded.json").read_text())["excluded_from_use"]
usable = sorted((int(s[1:]) for s in split["train"] if s not in excluded))
val = sorted(int(i) for i in np.random.default_rng(42).choice(usable, size=2, replace=False))
train = [i for i in usable if i not in val]
OUT.write_text(json.dumps({
    "method": "numpy.default_rng(42).choice(sorted usable train ids, size=2, replace=False), integer-sorted ids",
    "numpy_version": np.__version__,
    "train": [f"S{i}" for i in train],
    "validation": [f"S{i}" for i in val],
}, indent=2) + "\n")
print(OUT.read_text())
