"""Pick the 5 held-out subjects once (CLAUDE.md: seed 42, integer-sorted IDs).

Refuses to overwrite an existing splits.json. Opens no data files.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "splits.json"
IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

if OUT.exists():
    sys.exit(f"{OUT} already exists; the split is frozen. Not overwriting.")

sorted_ids = sorted(IDS)
held = sorted(int(i) for i in np.random.default_rng(42).choice(sorted_ids, size=5, replace=False))
train = [i for i in sorted_ids if i not in held]
OUT.write_text(json.dumps({
    "method": "numpy.default_rng(42).choice(sorted_ids, size=5, replace=False), integer-sorted ids",
    "numpy_version": np.__version__,
    "train": [f"S{i}" for i in train],
    "held_out": [f"S{i}" for i in held],
}, indent=2) + "\n")
print(OUT.read_text())
