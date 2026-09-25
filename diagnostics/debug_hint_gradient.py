"""Does the answer loss PULL a slot's attention toward its own hint row?
Gradient of the loss w.r.t. the pre-softmax cross-attention logits of the owning
slot, summed over all 8 iterations and heads, per token of its own sum / prod /
hint rows and of other pairs' hint rows.  Negative = raising that logit would
lower the loss (pull).  Broken pairs only (the ones that need the hint)."""
import os, sys, math
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
run = sys.argv[1] if len(sys.argv) > 1 else "seeds/route_seed0"
point = sys.argv[2] if len(sys.argv) > 2 else "1000"
sys.argv = [sys.argv[0], run, point]
src = open(os.path.join(HERE, "debug_encoder.py")).read().split("eps_tr, eps_te = episodes(777, 600)")[0]
ns = {"__file__": os.path.join(HERE, "debug_encoder.py")}
exec(compile(src, "debug_encoder_head", "exec"), ns)
torch, r, model, collate, episodes, row_types = (ns[k] for k in ("torch", "r", "model", "collate", "episodes", "row_types"))
F = torch.nn.functional
store = []
orig = model.cross_attn.forward
def wrapped(q_in, kv_in, mask=None, need_probs=False, **kw):
    out, probs = orig(q_in, kv_in, mask, need_probs=True, **kw)
    probs.retain_grad(); store.append(probs)
    return out, (probs if need_probs else None)
model.cross_attn.forward = wrapped
acc, cnt = defaultdict(float), defaultdict(int)
for chunk_i, (_, s0) in enumerate([(0, None)] * 4):
    eps = episodes(1234 + chunk_i, 22)
    chunk = [s for _, s in eps if s.meta["split_pairs"]][:48]
    store.clear(); model.zero_grad(set_to_none=True)
    batch = collate(chunk, r.device)
    out = model(batch, n_iters=8, collect=True)
    V = out.logits.shape[-1]
    F.cross_entropy(out.logits.reshape(-1, V), batch["target_ids"].reshape(-1), ignore_index=-100).backward()
    Bm = out.telemetry["bind_matrix"].float().cpu()
    glog = 0
    for p in store[1:]:                                   # skip the warm-init read (no grad of interest)
        g = p.grad
        glog = glog + (p * (g - (p * g).sum(-1, keepdim=True))).sum(1).float().cpu()   # [B,S,L], summed over heads
    for j, s in enumerate(chunk):
        toks = s.tokens; q = toks.index("Q"); kind, span = row_types(toks)
        for a_var, b_var in s.meta["split_pairs"]:
            pair = int(a_var[1:]) // 2
            owner = int(Bm[j, 2 * pair].argmax())
            for k in range(q):
                if not toks[k].startswith("X") or kind[k] not in ("sum", "prod", "hint"):
                    continue
                a0, b0 = span[k]
                own = int(toks[k][1:]) // 2 == pair
                key = (kind[k], "own" if own else "other")
                if toks[k] == a_var or not own:
                    acc[key] += float(glog[j, owner, a0:b0].sum()) / (b0 - a0)
                    cnt[key] += 1
print(f"[{run} @ {point}] dLoss/dlogit per token for the owning slot of each BROKEN pair (negative = pull)")
for rt in ("sum", "prod", "hint"):
    print(f"  {rt:5s} rows: own pair {acc[(rt, 'own')] / cnt[(rt, 'own')]:+.2e}   other pairs {acc[(rt, 'other')] / cnt[(rt, 'other')]:+.2e}")
