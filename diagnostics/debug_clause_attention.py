"""Owning slot's clause attention (final iteration, head mean) on its own pair's
sum / prod / hint clauses vs other pairs' clauses; uniform = 1/#clauses."""
import os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
run = sys.argv[1] if len(sys.argv) > 1 else "seeds/clause_seed0"
sys.argv = [sys.argv[0], run, sys.argv[2] if len(sys.argv) > 2 else "1000"]
src = open(os.path.join(HERE, "debug_encoder.py")).read().split("eps_tr, eps_te = episodes(777, 600)")[0]
ns = {"__file__": os.path.join(HERE, "debug_encoder.py")}
exec(compile(src, "debug_encoder_head", "exec"), ns)
torch, r, model, collate, episodes = (ns[k] for k in ("torch", "r", "model", "collate", "episodes"))
acc, cnt, nclause = defaultdict(float), defaultdict(int), []
with torch.no_grad():
    chunk = [s for _, s in episodes(1234, 64)]
    batch = collate(chunk, r.device)
    tel = model(batch, n_iters=8, collect=True).telemetry
    ca, Bm = tel["clause_attn"].float().cpu(), tel["bind_matrix"].float().cpu()
    seg = model.clause_sep_ids[batch["input_ids"]].long().cumsum(1).cpu()
for j, s in enumerate(chunk):
    toks = s.tokens; q = toks.index("Q")
    nclause.append(int(seg[j, len(toks) - 1]) + 1)
    kinds = {}
    for k in range(q):
        if toks[k] in ("PLUS", "TIMES", "GEQ") and toks[k - 1].startswith("X"):
            kinds[(int(toks[k - 1][1:]) // 2, {"PLUS": "sum", "TIMES": "prod", "GEQ": "hint"}[toks[k]])] = int(seg[j, k])
    for pair in range(8):
        owner = int(Bm[j, 2 * pair].argmax())
        for (pp, rt), c in kinds.items():
            key = (rt, "own" if pp == pair else "other")
            acc[key] += float(ca[j, owner, c]); cnt[key] += 1
print(f"[{run}] clause attention of the owning slot; uniform ~{1 / (sum(nclause) / len(nclause)):.3f} per clause")
for rt in ("sum", "prod", "hint"):
    print(f"  {rt:5s} clause: own pair {acc[(rt, 'own')] / cnt[(rt, 'own')]:.3f}   other pairs {acc[(rt, 'other')] / cnt[(rt, 'other')]:.3f}")
