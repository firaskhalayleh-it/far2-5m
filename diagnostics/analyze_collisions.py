"""Is the binding failure a SLOT COLLISION — variables of different pairs owning
the same workspace slot, so the decoder reads a slot that holds someone else's
value?  Uses the model's own ownership matrix B[variable, slot] (telemetry),
raw weights, T=8, fresh-ID symbreak.  Symmetric-pair answers only.
"""

import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repro_1000 as r
RUNS = r.RUNS  # noqa: E402
import torch  # noqa: E402
from far2.config import ModelConfig  # noqa: E402
from far2.data.base import collate, randomize_family_token  # noqa: E402
from far2.data.registry import ALL_GENERATORS  # noqa: E402
from far2.models import build_model  # noqa: E402

point = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
run = sys.argv[2] if len(sys.argv) > 2 else "run1000"
ckpt = torch.load(os.path.join(run if os.path.isabs(run) else os.path.join(RUNS, run), f"ckpt_step{point}.pt"),
                  map_location="cpu", weights_only=False)
cfg = ModelConfig(**ckpt["model_cfg"])
model = build_model(cfg)
model.load_state_dict(ckpt["model"], strict=False)
model.to(r.device).eval()
model.set_unlocked(set())

gen = ALL_GENERATORS["symbreak"]
acc = {"exclusive": [0, 0], "shared": [0, 0]}
tenant_hit, tenant_n = 0, 0
slots_used, peak, partner_same = [], [], [0, 0]
first = True
with torch.no_grad():
    for d in (1, 2, 4):
        rng = random.Random(f"1234:symbreak:{d}")
        for _ in range(8):
            samples = [randomize_family_token(gen.sample(rng, d), rng) for _ in range(64)]
            out = model(collate(samples, r.device), n_iters=8, collect=True)
            Bm = out.telemetry["bind_matrix"]
            Bm = Bm if torch.is_tensor(Bm) else torch.tensor(Bm)
            Bm = Bm.float().cpu()                                   # [B, V, S]
            if first:
                print("bind_matrix shape", tuple(Bm.shape), "| registers",
                      out.telemetry.get("n_registers"), "| slots", cfg.max_slots,
                      "init", cfg.init_slots)
                first = False
            owner = Bm.argmax(-1)                                   # [B, V]
            pred = out.logits.argmax(-1).tolist()
            for i, (s, row) in enumerate(zip(samples, pred)):
                q = s.tokens.index("Q")
                qvars = s.tokens[q + 1:q + 9]
                broken = {v for pair in s.meta["split_pairs"] for v in pair}
                vals = s.meta["vars"]
                own = owner[i].tolist()
                slots_used.append(len({own[2 * k] for k in range(8)}))
                peak.append(float(Bm[i].max(-1).values.mean()))
                for k in range(8):
                    partner_same[0] += own[2 * k] == own[2 * k + 1]
                    partner_same[1] += 1
                for v, p, t in zip(qvars, row, s.targets):
                    if v in broken:
                        continue
                    vi = int(v[1:])
                    tenants = [u for u in range(16) if u // 2 != vi // 2 and own[u] == own[vi]]
                    kind = "shared" if tenants else "exclusive"
                    guess = r.VOCAB.decode([p])[0]
                    acc[kind][0] += guess == t
                    acc[kind][1] += 1
                    if tenants and guess != t:
                        tenant_n += 1
                        tenant_hit += guess in {f"D{vals[f'X{u}']}" for u in tenants}

print(f"checkpoint {point} ({run}), raw weights, T=8, symmetric-pair answers")
print(f"distinct owner slots used by the 8 pairs: mean {sum(slots_used) / len(slots_used):.2f} "
      f"(8 needed) | histogram {dict(sorted(Counter(slots_used).items()))}")
print(f"ownership sharpness, mean max_s B[v,s]: {sum(peak) / len(peak):.2f}")
print(f"both variables of a pair share one owner slot: {partner_same[0] / partner_same[1]:.1%}")
for kind in ("exclusive", "shared"):
    k, n = acc[kind]
    print(f"answer whose owner slot is {kind:9s}: accuracy {k / max(n, 1):6.1%}  "
          f"(n={n}, {n / (acc['exclusive'][1] + acc['shared'][1]):.0%} of answers)")
if tenant_n:
    print(f"wrong answers from a shared slot that equal a CO-TENANT's value: "
          f"{tenant_hit / tenant_n:.1%}  (n={tenant_n})")
