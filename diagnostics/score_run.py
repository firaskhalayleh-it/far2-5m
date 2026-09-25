"""Raw-weights score table (Z0/T1/T8) for a kept run directory."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import repro_1000 as r
RUNS = r.RUNS
import torch
from far2.config import ModelConfig
from far2.models import build_model
run = sys.argv[1]
print(f"{'update':>6} {'T':>2} | {'token':>6} {'exact':>6} | {'symmetric':>9} {'broken':>7}   [{run}] raw weights")
for point in r.POINTS:
    ckpt = torch.load(os.path.join(RUNS, run, f"ckpt_step{point}.pt"), map_location="cpu", weights_only=False)
    model = build_model(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(r.device).eval(); model.set_unlocked(set())
    for T in (0, 1, 8):
        tok, ex, sym, brk, _ = r.score(model, T)
        print(f"{point:>6} {T:>2} | {tok:6.1%} {ex:6.1%} | {sym:9.1%} {brk:7.1%}", flush=True)
