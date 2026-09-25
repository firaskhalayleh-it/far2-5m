"""After the registered curve: score RAW and EMA weights of every checkpoint in a
run directory on the same fresh-ID symbreak cells (Z0/T1/T8), split by answer
kind.  The curve script reports EMA only; early EMA checkpoints are mostly init."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import repro_1000 as r
import torch
from far2.config import ModelConfig
from far2.models import build_model
run = sys.argv[1]
points = sorted(int(f[9:-3]) for f in os.listdir(run) if f.startswith("ckpt_step"))
print(f"{'update':>6} {'weights':>7} {'T':>2} | {'token':>6} {'exact':>6} | {'symmetric':>9} {'broken':>7}")
for point in points:
    ckpt = torch.load(os.path.join(run, f"ckpt_step{point}.pt"), map_location="cpu", weights_only=False)
    for label in ("ema", "model"):
        model = build_model(ModelConfig(**ckpt["model_cfg"]))
        model.load_state_dict(ckpt[label], strict=False)
        model.to(r.device).eval(); model.set_unlocked(set())
        for T in (0, 1, 8):
            tok, ex, sym, brk, _ = r.score(model, T)
            print(f"{point:>6} {'EMA' if label == 'ema' else 'raw':>7} {T:>2} | {tok:6.1%} {ex:6.1%} | {sym:9.1%} {brk:7.1%}", flush=True)
