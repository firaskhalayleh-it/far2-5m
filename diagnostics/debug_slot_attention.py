"""Per owning slot: does its cross-attention (final iteration, head mean) land on
its OWN pair's rows, or spread over everything?  Reported as attention mass per
token on own-pair sum/prod/hint rows vs other pairs' same-type rows, plus the
normalized entropy of each slot's distribution (1.0 = uniform)."""
import os, sys, math
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
run = sys.argv[1] if len(sys.argv) > 1 else "run1000_capped32"
sys.argv = [sys.argv[0], run, "1000"]
src = open(os.path.join(HERE, "debug_encoder.py")).read().split("eps_tr, eps_te = episodes(777, 600)")[0]
ns = {"__file__": os.path.join(HERE, "debug_encoder.py")}
exec(compile(src, "debug_encoder_head", "exec"), ns)
torch, r, model, collate, episodes, row_types = (ns[k] for k in ("torch", "r", "model", "collate", "episodes", "row_types"))
acc = defaultdict(lambda: [0.0, 0])
ents = []
with torch.no_grad():
    eps = episodes(1234, 128)
    for b in range(0, len(eps), 64):
        chunk = [s for _, s in eps[b:b + 64]]
        tel = model(collate(chunk, r.device), n_iters=8, collect=True).telemetry
        ca, Bm = tel["cross_attn"].float().cpu(), tel["bind_matrix"].float().cpu()
        for j, s in enumerate(chunk):
            toks = s.tokens; n = len(toks); q = toks.index("Q")
            kind, span = row_types(toks)
            # row spans per pair and type
            rows = defaultdict(list)
            for k in range(q):
                if toks[k].startswith("X") and kind[k] in ("sum", "prod", "hint"):
                    pair = int(toks[k][1:]) // 2
                    rows[(pair, kind[k])].append(span[k])
            for pair in range(8):
                owner = int(Bm[j, 2 * pair].argmax())
                dist = ca[j, owner, :n]
                ents.append(float(-(dist * dist.clamp_min(1e-12).log()).sum()) / math.log(n))
                for (pp, rt), spans in rows.items():
                    a0, b0 = spans[0]
                    m = float(dist[a0:b0].sum()) / (b0 - a0)
                    acc[(rt, "own" if pp == pair else "other")][0] += m
                    acc[(rt, "own" if pp == pair else "other")][1] += 1
print(f"[{run}] owning slot's final cross-attention; uniform would be ~{1 / 178:.4f} per token")
for rt in ("sum", "prod", "hint"):
    own, oth = (acc[(rt, w)][0] / acc[(rt, w)][1] for w in ("own", "other"))
    print(f"  {rt:5s} rows: own pair {own:.4f}/token   other pairs {oth:.4f}/token   ratio {own / oth:5.2f}x")
print(f"  normalized entropy of slot attention: mean {sum(ents) / len(ents):.3f} (1.0 = uniform)")
