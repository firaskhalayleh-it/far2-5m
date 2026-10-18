"""Parameter-matched Transformer (transformer_bind_match, 5.71M vs FAR2-bind 5.77M)
on the IDENTICAL contract: same seed and data stream, single-stage P3 mixture,
batch 128, same LR schedule, final-answer loss, stopped at 1,000 updates.
Raw weights scored on the same fresh-ID symbreak episodes."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import repro_1000 as r
RUNS = r.RUNS
import torch
from far2.config import TrainConfig
from far2.models import build_model
from far2.presets import get_preset
from far2.train import Stage, Trainer
from far2.utils import set_seed

out = os.path.join(RUNS, "run1000_transformer"); os.makedirs(out, exist_ok=True)
cfg = get_preset("transformer_bind_match")
tc = TrainConfig(steps=4000, batch_size=128, device="mps", seed=0, out_dir=out, warmup=266,
                 ema=0.999, ipt_alpha=0.0, cycle_supervision="final",
                 checkpoint_steps=r.POINTS, log_every=50)
set_seed(0)
tr = Trainer(build_model(cfg), cfg, tc, [Stage("only", 1.0, r.fc._mix(), depth="fixed")])
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
print(f"trained {calls[0]} updates in {(time.perf_counter() - t0) / 60:.1f} min "
      f"(params {tr.model.param_count():,})", flush=True)
print(f"{'update':>6} | {'token':>6} {'exact':>6} | {'symmetric':>9} {'broken':>7}   [transformer_bind_match] raw weights")
for point in r.POINTS:
    ckpt = torch.load(os.path.join(out, f"ckpt_step{point}.pt"), map_location="cpu", weights_only=False)
    model = build_model(cfg); model.load_state_dict(ckpt["model"], strict=False)
    model.to(r.device).eval()
    tok, ex, sym, brk, _ = r.score(model, None)
    print(f"{point:>6} | {tok:6.1%} {ex:6.1%} | {sym:9.1%} {brk:7.1%}", flush=True)
