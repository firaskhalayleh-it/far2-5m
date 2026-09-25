"""Same 1,000-update arm as repro_1000.py, but the checkpoints are kept on disk
(scratchpad/run1000) so later analyses cost seconds instead of a retrain."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repro_1000 as r
RUNS = r.RUNS  # noqa: E402

out = os.path.join(RUNS, os.environ.get("FAR2_RUN", "run1000"))
os.makedirs(out, exist_ok=True)
r.train(out)
print("checkpoints:", sorted(f for f in os.listdir(out) if f.endswith(".pt")), flush=True)
