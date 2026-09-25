"""What exactly does the 1,000-update model get wrong on symbreak?

Raw weights, T=8, fresh-ID episodes.  For every symmetric-pair answer (true
value a, the sum row shows 2a zero-padded to two digits):
  * accuracy by true value a  -> is one value class failing (e.g. sums >= 10)?
  * for each error, is the prediction the value of ANOTHER variable in the same
    episode (retrieval/binding error: right rule, wrong row) or a value present
    nowhere in the episode (arithmetic error: right row, wrong rule)?
A chance-level guesser lands on another variable's value often, so the
binding share is compared with that expectation.
"""

import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repro_1000 as r
RUNS = r.RUNS  # noqa: E402  (sets up repo paths)
import torch  # noqa: E402
from far2.data.base import collate, randomize_family_token  # noqa: E402
from far2.data.registry import ALL_GENERATORS  # noqa: E402
from far2.models import build_model  # noqa: E402
from far2.config import ModelConfig  # noqa: E402

point = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
ckpt = torch.load(os.path.join(RUNS, (sys.argv[2] if len(sys.argv) > 2 else "run1000"), f"ckpt_step{point}.pt"),
                  map_location="cpu", weights_only=False)
cfg = ModelConfig(**ckpt["model_cfg"])
model = build_model(cfg)
model.load_state_dict(ckpt["model"], strict=False)  # RAW weights
model.to(r.device).eval()
model.set_unlocked(set())

gen = ALL_GENERATORS["symbreak"]
by_value = defaultdict(lambda: [0, 0])
by_rank = defaultdict(lambda: [0, 0])
kinds, expected_other = Counter(), []
per_episode_right = Counter()
with torch.no_grad():
    for d in (1, 2, 4):
        rng = random.Random(f"1234:symbreak:{d}")
        for _ in range(8):
            samples = [randomize_family_token(gen.sample(rng, d), rng) for _ in range(64)]
            pred = model(collate(samples, r.device), n_iters=8).logits.argmax(-1).tolist()
            for s, row in zip(samples, pred):
                q = s.tokens.index("Q")
                qvars = s.tokens[q + 1:q + 9]
                broken = {v for pair in s.meta["split_pairs"] for v in pair}
                values = s.meta["vars"]
                right = 0
                for rank, (v, p, t) in enumerate(zip(qvars, row, s.targets)):
                    guess = r.VOCAB.decode([p])[0]
                    good = guess == t
                    right += good
                    if v in broken:
                        continue
                    a = int(t[1:])
                    by_value[a][0] += good
                    by_value[a][1] += 1
                    by_rank[rank][0] += good
                    by_rank[rank][1] += 1
                    if not good:
                        others = {f"D{x}" for k, x in values.items() if k != v} - {t}
                        kinds["another variable's value" if guess in others
                              else "a value not in the episode"] += 1
                        expected_other.append(len(others) / 9)  # 9 wrong digits to land on
                per_episode_right[right] += 1

print(f"checkpoint {point}, raw weights, T=8 — symmetric-pair answers only")
print("accuracy by true value a (sum row shows 2a as two digits):")
for a in range(10):
    k, n = by_value[a]
    print(f"  a={a}  sum={2 * a:02d}  {k / n:6.1%}  (n={n})")
print("accuracy by answer position (rank in the query):")
print("  " + "  ".join(f"#{k}:{by_rank[k][0] / by_rank[k][1]:.0%}" for k in range(8)))
tot = sum(kinds.values())
print(f"errors on symmetric answers: {tot}")
for k, n in kinds.most_common():
    print(f"  predicted {k}: {n / tot:.1%}")
print(f"  (a blind wrong guess would land on another variable's value "
      f"{sum(expected_other) / len(expected_other):.1%} of the time)")
print("answers right per episode (of 8):",
      {k: per_episode_right[k] for k in sorted(per_episode_right)})
