"""Isolated encoder bench: which encoder DESIGN can learn the relation the slot needs?

Only the encoder (embeddings, optional local mix / rope / QK-norm, encoder
blocks, final LN) plus two linear readouts are trained, with the main run's
optimizer settings (AdamW lr 3e-4, betas 0.9/0.95, wd 0.01, grad clip 0.5).
Targets, broken pairs of fresh symbreak episodes (R in {1,2,4}):
  local     at the variable's hint-row token: "left of GEQ"
  crossrow  at the variable's SUM-row token: "this variable holds the larger
            root" (the hint sits in another row ~100 tokens away)
The readouts are supervised: this measures what each design CAN learn and how
fast, not a training recipe for the model.  Test = project eval seed 1234.
"""

import random
import sys
import time
from dataclasses import replace

import os  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from far2.data import GENERATORS, collate  # noqa: E402
from far2.data.base import randomize_family_token  # noqa: E402
from far2.models import build_model  # noqa: E402
from far2.presets import get_preset  # noqa: E402

dev = torch.device("mps")
gen = GENERATORS["symbreak"]
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
BATCH = 64


def targets(s):
    toks = s.tokens
    q = toks.index("Q")
    local, cross = [], []
    for a_var, b_var in s.meta["split_pairs"]:
        for v in (a_var, b_var):
            pos = [k for k in range(q) if toks[k] == v]
            hint = [k for k in pos if toks[k + 1] == "GEQ" or toks[k - 1] == "GEQ"][0]
            summ = [k for k in pos if toks[k + 1] == "PLUS" or toks[k - 1] == "PLUS"][0]
            local.append((hint, float(toks[hint + 1] == "GEQ")))
            partner = f"X{int(v[1:]) ^ 1}"
            cross.append((summ, float(s.meta["vars"][v] > s.meta["vars"][partner])))
    return local, cross


def make_batch(rng, n):
    samples = []
    while len(samples) < n:
        s = gen.sample(rng, rng.choice([1, 2, 4]))
        if s.meta["split_pairs"]:
            samples.append(randomize_family_token(s, rng))
    b = collate(samples, dev)
    idx = {"local": ([], [], []), "cross": ([], [], [])}
    for j, s in enumerate(samples):
        loc, cr = targets(s)
        for key, rows in (("local", loc), ("cross", cr)):
            for k, y in rows:
                idx[key][0].append(j)
                idx[key][1].append(k)
                idx[key][2].append(y)
    return b, {k: (torch.tensor(a, device=dev), torch.tensor(p, device=dev),
                   torch.tensor(y, device=dev)) for k, (a, p, y) in idx.items()}


test_rng = random.Random("1234:encoder-bench")
TEST = [make_batch(test_rng, 64) for _ in range(4)]

base = replace(get_preset("far2_bind"), n_iters_train=8, init_slots=14,
               workspace_init="input", bind_norm="capped")
VARIANTS = {
    "baseline (1 layer, abs pos)": {},
    "rope": dict(enc_pos="rope"),
    "local mix": dict(enc_local_mix=True),
    "qk-norm": dict(enc_qk_norm=True),
    "rope + qk-norm": dict(enc_pos="rope", enc_qk_norm=True),
    "local mix + qk-norm": dict(enc_local_mix=True, enc_qk_norm=True),
    "local mix + rope + qk-norm": dict(enc_local_mix=True, enc_pos="rope", enc_qk_norm=True),
    "2 layers": dict(n_encoder_layers=2),
    "2 layers + rope + qk-norm": dict(n_encoder_layers=2, enc_pos="rope", enc_qk_norm=True),
}
if len(sys.argv) > 2:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in sys.argv[2].split(",")}


def encoder_params(m):
    keep = ("tok.", "pos.", "sig.", "encoder.", "enc_ln.", "local_mix.", "clause_")
    return [p for n, p in m.named_parameters() if n.startswith(keep)]


@torch.no_grad()
def evaluate(m, heads):
    hit = {"local": [0, 0], "cross": [0, 0]}
    for b, idx in TEST:
        H, _ = m._encode(b)
        for key, (bi, pi, y) in idx.items():
            pred = heads[key](H[bi, pi]).squeeze(-1) > 0
            hit[key][0] += int((pred.float() == y).sum())
            hit[key][1] += len(y)
    return {k: a / n for k, (a, n) in hit.items()}


print(f"encoder bench: {STEPS} steps, batch {BATCH}, lr 3e-4, clip 0.5 (chance 50%)")
print(f"{'variant':30s} {'enc params':>10} | " + "  ".join(f"{'@' + str(s):>13}" for s in (100, 300, STEPS))
      + "   (local / crossrow)")
for name, kw in VARIANTS.items():
    torch.manual_seed(0)
    m = build_model(replace(base, **kw)).to(dev).train()
    heads = torch.nn.ModuleDict({"local": torch.nn.Linear(base.d_model, 1),
                                 "cross": torch.nn.Linear(base.d_model, 1)}).to(dev)
    params = encoder_params(m) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(f"train:{name}")
    marks, t0 = [], time.perf_counter()
    for step in range(1, STEPS + 1):
        b, idx = make_batch(rng, BATCH)
        H, _ = m._encode(b)
        loss = sum(F.binary_cross_entropy_with_logits(heads[k](H[bi, pi]).squeeze(-1), y)
                   for k, (bi, pi, y) in idx.items())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()
        if step in (100, 300, STEPS):
            m.eval()
            marks.append(evaluate(m, heads))
            m.train()
    n_enc = sum(p.numel() for p in encoder_params(m))
    print(f"{name:30s} {n_enc:>10,} | " + "  ".join(f"{a['local']:5.1%} / {a['cross']:5.1%}" for a in marks)
          + f"   [{time.perf_counter() - t0:.0f}s]", flush=True)
