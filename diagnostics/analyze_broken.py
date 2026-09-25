"""Where do broken-pair answers go wrong, and does accuracy keep rising with T?

A broken pair (a != b) is queried on one side.  The answer needs: find the
pair's sum row and product row, solve for the two roots, find the late hint
row (hi GEQ lo), and pick the root on the queried side.  Each wrong answer is
classified by what the model returned instead:
  partner   the other root of the same pair -> roots right, SELECTION wrong
  half-sum  (a+b)//2                         -> pair treated as symmetric
  other     a value of another pair          -> bound to the wrong pair
  none      a value present nowhere          -> arithmetic
Raw weights, fresh-ID symbreak R in {1,2,4} (project eval seed 1234).
"""

import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repro_1000 as r  # noqa: E402
import torch  # noqa: E402
from far2.config import ModelConfig  # noqa: E402
from far2.data.base import collate, randomize_family_token  # noqa: E402
from far2.data.registry import ALL_GENERATORS  # noqa: E402
from far2.models import build_model  # noqa: E402

run = sys.argv[1]
point = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
path = run if os.path.isabs(run) else os.path.join(r.RUNS, run)
ckpt = torch.load(os.path.join(path, f"ckpt_step{point}.pt"), map_location="cpu",
                  weights_only=False)
cfg = ModelConfig(**ckpt["model_cfg"])
model = build_model(cfg)
model.load_state_dict(ckpt["model"], strict=False)
model.to(r.device).eval()
if hasattr(model, "set_unlocked"):
    model.set_unlocked(set(os.environ.get("FAR2_UNLOCK", "").split(",")) - {""})
gen = ALL_GENERATORS["symbreak"]
recurrent = cfg.kind == "far2"


@torch.no_grad()
def classify(T):
    kinds, by_r = Counter(), defaultdict(lambda: [0, 0])
    sym, brk, exact, n_ep = [0, 0], [0, 0], 0, 0
    for d in (1, 2, 4):
        rng = random.Random(f"1234:symbreak:{d}")
        for _ in range(8):
            samples = [randomize_family_token(gen.sample(rng, d), rng) for _ in range(64)]
            out = model(collate(samples, r.device), n_iters=T)
            for s, row in zip(samples, out.logits.argmax(-1).tolist()):
                q = s.tokens.index("Q")
                qvars = s.tokens[q + 1:q + 9]
                vals = s.meta["vars"]
                broken = {v for pair in s.meta["split_pairs"] for v in pair}
                ok_all = True
                for v, p, t in zip(qvars, row, s.targets):
                    guess = r.VOCAB.decode([p])[0]
                    good = guess == t
                    ok_all &= good
                    if v not in broken:
                        sym[0] += good
                        sym[1] += 1
                        continue
                    brk[0] += good
                    brk[1] += 1
                    by_r[d][0] += good
                    by_r[d][1] += 1
                    if good:
                        kinds["correct"] += 1
                        continue
                    i = int(v[1:])
                    partner = f"X{i ^ 1}"
                    x, y = vals[v], vals[partner]
                    others = {f"D{vals[k]}" for k in vals if k not in (v, partner)}
                    if guess == f"D{y}":
                        kinds["partner (selection)"] += 1
                    elif guess == f"D{(x + y) // 2}":
                        kinds["half-sum (treated symmetric)"] += 1
                    elif guess in others:
                        kinds["other pair (binding)"] += 1
                    else:
                        kinds["none (arithmetic)"] += 1
                exact += ok_all
                n_ep += 1
    return kinds, by_r, sym, brk, exact / n_ep


iters = (0, 1, 2, 4, 8, 12, 16, 32) if recurrent else (None,)
print(f"[{run} @ {point}] raw weights, symbreak fresh-ID R in {{1,2,4}}, 768 episodes")
print(f"{'T':>4} | {'exact':>6} {'symmetric':>9} {'broken':>7}")
detail = None
for T in iters:
    kinds, by_r, sym, brk, ex = classify(T)
    print(f"{str(T):>4} | {ex:6.1%} {sym[0] / sym[1]:9.1%} {brk[0] / brk[1]:7.1%}", flush=True)
    if T in (8, None):
        detail = (kinds, by_r, brk)
kinds, by_r, brk = detail
print(f"\nbroken-pair answers at T={8 if recurrent else 'n/a'} (n={brk[1]}):")
for k in ("correct", "partner (selection)", "half-sum (treated symmetric)",
          "other pair (binding)", "none (arithmetic)"):
    print(f"  {k:30s} {kinds[k] / brk[1]:6.1%}")
print("  roots found (correct + partner):  "
      f"{(kinds['correct'] + kinds['partner (selection)']) / brk[1]:6.1%}")
print("  selection correct/(correct+partner):  "
      f"{kinds['correct'] / max(kinds['correct'] + kinds['partner (selection)'], 1):6.1%}")
print("broken accuracy by number of broken pairs R: " +
      "  ".join(f"R={d}: {k / n:.1%}" for d, (k, n) in sorted(by_r.items())))
