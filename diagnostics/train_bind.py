"""Prediction test: if BINDING is what stalls the 1,000-update arm, then switching
on the repo's existing entity/binding curriculum (bind_weight=0.5; labels shape
training geometry only and never enter the forward pass) should lift
symmetric-pair accuracy far above the ~59% plateau at the same 1,000 updates,
and errors should stop being "another variable's value".  Everything else is
identical to the baseline arm (same seed, data stream, schedule, T=8)."""

import os
import sys
import time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repro_1000 as r
RUNS = r.RUNS  # noqa: E402
import torch  # noqa: E402
from far2.config import TrainConfig  # noqa: E402
from far2.models import build_model  # noqa: E402
from far2.presets import get_preset  # noqa: E402
from far2.train import Stage, Trainer  # noqa: E402
from far2.utils import set_seed  # noqa: E402

out = os.path.join(RUNS, "run1000_bind")
os.makedirs(out, exist_ok=True)
cfg = replace(get_preset("far2_bind"), n_iters_train=8, init_slots=14,
              workspace_init="input")
tc = TrainConfig(steps=4000, batch_size=128, device="mps", seed=0, out_dir=out,
                 warmup=266, ema=0.999, ipt_alpha=0.0, cycle_supervision="final",
                 checkpoint_steps=r.POINTS, log_every=50)
set_seed(0)
tr = Trainer(build_model(cfg), cfg, tc,
             [Stage("only", 1.0, r.fc._mix(), depth="fixed", bind_weight=0.5)])
inner, calls = tr.opt.step, [0]


def counted(*a, **k):
    if calls[0] == r.STOP:
        raise r.Stop
    calls[0] += 1
    return inner(*a, **k)


tr.opt.step = counted
t0 = time.perf_counter()
try:
    tr.train()
except r.Stop:
    pass
print(f"trained {calls[0]} updates with bind_weight=0.5 in "
      f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)

print(f"\n{'update':>6} {'T':>2} | {'token':>6} {'exact':>6} | {'symmetric':>9} {'broken':>7}"
      "   raw weights, bind labels OFF at eval (they never enter forward)")
for point in r.POINTS:
    ckpt = torch.load(os.path.join(out, f"ckpt_step{point}.pt"),
                      map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(r.device).eval()
    model.set_unlocked(set())
    for T in (0, 1, 8):
        tok, ex, sym, brk, _ = r.score(model, T)
        print(f"{point:>6} {T:>2} | {tok:6.1%} {ex:6.1%} | {sym:9.1%} {brk:7.1%}", flush=True)
