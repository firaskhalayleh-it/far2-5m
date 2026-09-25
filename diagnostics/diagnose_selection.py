"""Step-1 diagnostics for the broken-pair SELECTION failure (eval only).

1. Canonical-order bias: split broken accuracy by whether the queried variable
   is the pair's first (even-index) variable and whether its value is the
   larger root.  ~100% on one diagonal and ~0% on the other = fixed ordering.
2. Linear probe for the bit "the pair's first variable holds the larger root"
   from the state of the slot that owns the first variable (T=0 and T=8), plus
   a pooled-workspace probe and a shuffled-label control.  Trained on fresh
   episodes (seed 777), tested on the project's eval episodes (seed 1234).
   Chance ~ the majority rate.  At chance -> the bit never enters the slot
   (encoding/binding).  Decodable but outputs 50/50 -> readout problem.
3. Error categories at every T.
4. Choice stability: for the same broken answer, does correct-vs-partner flip
   across T in {4, 8, 12, 16}?
"""

import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repro_1000 as r  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from far2.config import ModelConfig  # noqa: E402
from far2.data.base import collate, randomize_family_token  # noqa: E402
from far2.data.registry import ALL_GENERATORS  # noqa: E402
from far2.models import build_model  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "run1000_capped32"
point = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
ckpt = torch.load(os.path.join(r.RUNS, run, f"ckpt_step{point}.pt"),
                  map_location="cpu", weights_only=False)
cfg = ModelConfig(**ckpt["model_cfg"])
model = build_model(cfg)
model.load_state_dict(ckpt["model"], strict=False)
model.to(r.device).eval()
model.set_unlocked(set(os.environ.get("FAR2_UNLOCK", "").split(",")) - {""})
gen = ALL_GENERATORS["symbreak"]
KINDS = ("correct", "partner", "half-sum", "other pair", "arithmetic")


def episodes(seed, per_d):
    out = []
    for d in (1, 2, 4):
        rng = random.Random(f"{seed}:symbreak:{d}")
        out += [(d, randomize_family_token(gen.sample(rng, d), rng)) for _ in range(per_d)]
    return out


@torch.no_grad()
def run_T(eps, T, want_states=False):
    """Per broken answer: (key, kind, queried_first, queried_larger); optional
    probe rows: (owner-slot state, pooled state, bit) per broken pair."""
    answers, rows = [], []
    for b in range(0, len(eps), 64):
        chunk = [s for _, s in eps[b:b + 64]]
        out = model(collate(chunk, r.device), n_iters=T, collect=True)
        pred = out.logits.argmax(-1).tolist()
        if want_states:
            Z = out.telemetry["Z_final"].float().cpu()
            Bm = out.telemetry["bind_matrix"].float().cpu()
            act = out.telemetry["active_mask"].float().cpu()
        for j, (s, row) in enumerate(zip(chunk, pred)):
            q = s.tokens.index("Q")
            qvars = s.tokens[q + 1:q + 9]
            vals = s.meta["vars"]
            broken = {v for pair in s.meta["split_pairs"] for v in pair}
            for v, p, t in zip(qvars, row, s.targets):
                if v not in broken:
                    continue
                i = int(v[1:])
                partner = f"X{i ^ 1}"
                x, y = vals[v], vals[partner]
                guess = r.VOCAB.decode([p])[0]
                others = {f"D{vals[k]}" for k in vals if k not in (v, partner)}
                if guess == t:
                    kind = "correct"
                elif guess == f"D{y}":
                    kind = "partner"
                elif guess == f"D{(x + y) // 2}":
                    kind = "half-sum"
                elif guess in others:
                    kind = "other pair"
                else:
                    kind = "arithmetic"
                answers.append(((b + j, v), kind, i % 2 == 0, x > y))
            if want_states:
                zs = Z[j]
                pooled = (zs * act[j][:, None]).sum(0) / act[j].sum()
                for a_var, b_var in s.meta["split_pairs"]:
                    ia = int(a_var[1:])
                    owner = int(Bm[j, ia].argmax())
                    rows.append((zs[owner], pooled, float(vals[a_var] > vals[b_var])))
    return answers, rows


def probe(train, test, shuffle=False):
    Xtr = torch.stack([row[0] for row in train]); ytr = torch.tensor([row[1] for row in train])
    Xte = torch.stack([row[0] for row in test]); yte = torch.tensor([row[1] for row in test])
    if shuffle:
        ytr = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(0))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=300)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + 1e-2 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float((((Xte @ w + b) > 0).float() == yte).float().mean())


test_eps = episodes(1234, 256)
train_eps = episodes(777, 1200)

# 3 + 4: categories at every T, and per-answer choices for flip analysis
print(f"[{run} @ {point}] raw weights\n\nerror categories of broken answers by T:")
print(f"{'T':>3} | " + " ".join(f"{k:>11}" for k in KINDS))
choice = defaultdict(dict)
for T in (0, 1, 2, 4, 8, 12, 16, 32):
    ans, _ = run_T(test_eps, T)
    c = Counter(k for _, k, _, _ in ans)
    print(f"{T:>3} | " + " ".join(f"{c[k] / len(ans):11.1%}" for k in KINDS), flush=True)
    for key, kind, _, _ in ans:
        choice[key][T] = kind
    if T == 8:
        ans8 = ans

# 1: canonical-order bias at T=8
cell = defaultdict(lambda: [0, 0])
for _, kind, first, larger in ans8:
    cell[(first, larger)][0] += kind == "correct"
    cell[(first, larger)][1] += 1
print("\nbroken accuracy at T=8 by queried side and root (n per cell):")
for first in (True, False):
    print("  queried = " + ("first (even) var " if first else "second (odd) var ") +
          "  ".join(f"{'larger' if lg else 'smaller'} root: {cell[(first, lg)][0] / cell[(first, lg)][1]:5.1%} "
                    f"(n={cell[(first, lg)][1]})" for lg in (True, False)))
sel = defaultdict(lambda: [0, 0])
for _, kind, first, larger in ans8:
    if kind in ("correct", "partner"):
        sel[larger][0] += kind == "correct"
        sel[larger][1] += 1
print("  among answers where the roots were found, correct choice when the queried value is "
      + "  ".join(f"{'larger' if lg else 'smaller'}: {sel[lg][0] / sel[lg][1]:5.1%}" for lg in (True, False)))

# 4: stability of correct-vs-partner across T
flip, stable = 0, 0
for key, by_t in choice.items():
    picks = [by_t[T] for T in (4, 8, 12, 16) if by_t.get(T) in ("correct", "partner")]
    if len(picks) == 4:
        if len(set(picks)) > 1:
            flip += 1
        else:
            stable += 1
print(f"\nchoice stability across T=4,8,12,16 (answers that are correct-or-partner at all four): "
      f"stable {stable / (flip + stable):.1%}, flips {flip / (flip + stable):.1%} (n={flip + stable})")

# 2: probes for the ordering bit
print("\nlinear probe for 'first variable holds the larger root' (test on eval episodes):")
for T in (0, 8):
    _, tr = run_T(train_eps, T, want_states=True)
    _, te = run_T(test_eps, T, want_states=True)
    maj = max(sum(x[2] for x in te), len(te) - sum(x[2] for x in te)) / len(te)
    owner = probe([(a, y) for a, _, y in tr], [(a, y) for a, _, y in te])
    pooled = probe([(p, y) for _, p, y in tr], [(p, y) for _, p, y in te])
    shuf = probe([(a, y) for a, _, y in tr], [(a, y) for a, _, y in te], shuffle=True)
    print(f"  T={T}: owner slot {owner:5.1%} | pooled workspace {pooled:5.1%} | "
          f"shuffled-label control {shuf:5.1%} | majority {maj:5.1%} "
          f"(train {len(tr)} pairs, test {len(te)})", flush=True)
