"""Upper-bound diagnostic: does giving each variable of a broken pair its own
slot fix root SELECTION?  Same contract as run1000_capped32 (capped x32, warm
init, T=8, final-answer loss, NO binding labels, single-stage P3 mixture,
1,000 updates) except: 8 concept slots (one per pair) and SPLIT forced by the
task condition on every broken pair (repr_policy=oracle, semantic split).
Ground truth decides WHEN to split, so this is a bound, never a result."""
import os, sys, time
from dataclasses import replace
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import repro_1000 as r
from far2.config import TrainConfig
from far2.models import build_model
from far2.presets import get_preset
from far2.train import Stage, Trainer
from far2.utils import set_seed
out = os.path.join(r.RUNS, "run1000_oracle_split"); os.makedirs(out, exist_ok=True)
cfg = replace(get_preset("far2_bind"), n_iters_train=8, init_slots=8, workspace_init="input",
              bind_norm="capped", bind_sweeps=32, repr_policy="oracle", oracle_action="split")
tc = TrainConfig(steps=4000, batch_size=128, device="mps", seed=0, out_dir=out, warmup=266, ema=0.999,
                 ipt_alpha=0.0, cycle_supervision="final", checkpoint_steps=r.POINTS, log_every=50)
set_seed(0)
tr = Trainer(build_model(cfg), cfg, tc, [Stage("only", 1.0, r.fc._mix(), unlocked={"expand", "split"},
                                               depth="fixed", bind_weight=0.0)])
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
print(f"trained {calls[0]} updates, oracle split, in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
