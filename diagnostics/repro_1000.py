"""Reproduce the early-stopped 1,000-update arm and ask two questions of it.

1. Did the evaluation hide learning?  Each checkpoint stores both the raw
   weights and the EMA(0.999, no bias correction) weights; the project evaluates
   EMA only.  Score both on identical fresh-ID symbreak episodes.
2. WHAT was learned?  Split per-answer accuracy into symmetric pairs (answer is
   s/2, a lookup) and broken pairs (roots of x^2 - s x + p chosen by a late hint).

Same arm contract as the old run: far2_bind fixed14, input-warm, T=8, batch 128,
final-answer loss only, no binding labels, single-stage P3 mixture, the LR
schedule of a 4,000-step run stopped at 1,000.  Writes only to a temp directory.
"""

import os
import random
import sys
import tempfile
import time
from dataclasses import replace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs_diag")  # gitignored (runs_*/)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

import torch  # noqa: E402

import run_foundation_curve as fc  # noqa: E402
from far2.config import TrainConfig  # noqa: E402
from far2.data.base import collate, randomize_family_token  # noqa: E402
from far2.data.registry import ALL_GENERATORS  # noqa: E402
from far2.data.vocab import VOCAB  # noqa: E402
from far2.models import build_model  # noqa: E402
from far2.presets import get_preset  # noqa: E402
from far2.train import Stage, Trainer  # noqa: E402
from far2.utils import set_seed  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sinkhorn_patch  # noqa: E402,F401  (no-op unless FAR2_SINKHORN=1)

STOP, POINTS, N_EVAL = 1000, (250, 500, 1000), 256
device = torch.device("mps")


class Stop(Exception):
    pass


def train(out):
    cfg = replace(get_preset("far2_bind"), n_iters_train=8, init_slots=14,
                  workspace_init="input")
    tc = TrainConfig(steps=4000, batch_size=128, device="mps", seed=0, out_dir=out,
                     warmup=266, ema=0.999, ipt_alpha=0.0, cycle_supervision="final",
                     checkpoint_steps=POINTS, log_every=50)
    set_seed(0)
    tr = Trainer(build_model(cfg), cfg, tc,
                 [Stage("only", 1.0, fc._mix(), depth="fixed", bind_weight=0.0)])
    inner, calls = tr.opt.step, [0]

    def counted(*a, **k):
        if calls[0] == STOP:  # checkpoint for update STOP was written last iteration
            raise Stop
        calls[0] += 1
        return inner(*a, **k)

    tr.opt.step = counted
    t0 = time.perf_counter()
    try:
        tr.train()
    except Stop:
        pass
    print(f"trained {calls[0]} updates in {(time.perf_counter() - t0) / 60:.1f} min "
          f"(params {tr.model.param_count():,})", flush=True)
    return cfg


@torch.no_grad()
def score(model, T):
    """Fresh-ID symbreak R in {1,2,4}: token, exact, and accuracy by answer kind."""
    gen = ALL_GENERATORS["symbreak"]
    hit = {"sym": [0, 0], "broken": [0, 0]}
    exact = [0, 0]
    for d in fc.ID_SYMBREAK:
        rng = random.Random(f"1234:symbreak:{d}")  # the project's paired eval seed
        left = N_EVAL
        while left > 0:
            n = min(64, left)
            left -= n
            samples = [randomize_family_token(gen.sample(rng, d), rng) for _ in range(n)]
            out = model(collate(samples, device), n_iters=T)
            pred = out.logits.argmax(-1).tolist()
            for s, row in zip(samples, pred):
                q = s.tokens.index("Q")
                qvars = s.tokens[q + 1:q + 1 + len(s.targets)]
                broken = {v for pair in s.meta["split_pairs"] for v in pair}
                ok = [VOCAB.decode([p])[0] == t for p, t in zip(row, s.targets)]
                for v, good in zip(qvars, ok):
                    cell = hit["broken" if v in broken else "sym"]
                    cell[0] += good
                    cell[1] += 1
                exact[0] += all(ok)
                exact[1] += 1
    tok = (hit["sym"][0] + hit["broken"][0]) / (hit["sym"][1] + hit["broken"][1])
    return (tok, exact[0] / exact[1], hit["sym"][0] / hit["sym"][1],
            hit["broken"][0] / max(hit["broken"][1], 1), hit["broken"][1])


def main():
    with tempfile.TemporaryDirectory() as out:
        cfg = train(out)
        print(f"\n{'update':>6} {'weights':>7} {'T':>2} | {'token':>6} {'exact':>6} | "
              f"{'symmetric':>9} {'broken':>7}   (chance 10%)")
        for point in POINTS:
            ckpt = torch.load(os.path.join(out, f"ckpt_step{point}.pt"),
                              map_location="cpu", weights_only=False)
            for label, weights in (("EMA", ckpt["ema"]), ("raw", ckpt["model"])):
                model = build_model(cfg)
                model.load_state_dict(weights, strict=False)
                model.to(device).eval()
                model.set_unlocked(set())
                for T in (0, 1, 8):
                    tok, ex, sym, brk, nb = score(model, T)
                    print(f"{point:>6} {label:>7} {T:>2} | {tok:6.1%} {ex:6.1%} | "
                          f"{sym:9.1%} {brk:7.1%}", flush=True)
        print(f"(n = {3 * N_EVAL} episodes, {3 * N_EVAL * 8} answers, {nb} of them broken-pair)")


if __name__ == "__main__":
    main()
