"""Same trained weights, evaluation-only change: how many cap sweeps does the
ownership law need before it actually holds, and what does accuracy do then?"""
import os, random, sys
from dataclasses import replace
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import repro_1000 as r
import torch
from far2.config import ModelConfig
from far2.data.base import collate, randomize_family_token
from far2.data.registry import ALL_GENERATORS
from far2.models import build_model
run, point = sys.argv[1], int(sys.argv[2])
ckpt = torch.load(os.path.join(run, f"ckpt_step{point}.pt"), map_location="cpu", weights_only=False)
base = ModelConfig(**ckpt["model_cfg"])
gen = ALL_GENERATORS["symbreak"]
print(f"{'law':>14} | {'token':>6} {'exact':>6} {'symmetric':>9} {'broken':>7} | slots used  max slot mass  score range")
for norm, sweeps in (("softmax", 1), ("capped", 6), ("capped", 24), ("capped", 100), ("capped", 400)):
    cfg = replace(base, bind_norm=norm, bind_sweeps=sweeps)
    model = build_model(cfg); model.load_state_dict(ckpt["model"], strict=False)
    model.to(r.device).eval(); model.set_unlocked(set())
    tok, ex, sym, brk, _ = r.score(model, 8)
    used, mass = [], []
    with torch.no_grad():
        rng = random.Random("1234:symbreak:4")
        samples = [randomize_family_token(gen.sample(rng, 4), rng) for _ in range(64)]
        Bm = model(collate(samples, r.device), n_iters=8, collect=True).telemetry["bind_matrix"].float().cpu()
        own = Bm.argmax(-1)
        used = [len({int(own[i, 2 * k]) for k in range(8)}) for i in range(64)]
        mass = float(Bm.sum(1).max(-1).values.mean())
    print(f"{norm + ' x' + str(sweeps):>14} | {tok:6.1%} {ex:6.1%} {sym:9.1%} {brk:7.1%} | {sum(used)/64:9.2f}  {mass:13.2f}", flush=True)
