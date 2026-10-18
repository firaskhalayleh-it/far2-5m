"""Why does the routed slot ignore its own hint row?
Per head and per iteration: attention mass (per token) of the owning slot on its
own pair's sum / prod / hint rows, and the raw pre-softmax score the slot gives
the hint-row X token vs the sum-row X token of its own variable."""
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
calls = []
def hook(mod, inp, out):
    q_in, kv_in = inp[0], inp[1]
    B, Lq, _ = q_in.shape; Lk = kv_in.shape[1]
    q = mod.wq(q_in).view(B, Lq, mod.h, mod.dh).transpose(1, 2)
    k = mod.wk(kv_in).view(B, Lk, mod.h, mod.dh).transpose(1, 2)
    if mod.qk_scale is not None:
        s = (torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(k, dim=-1).transpose(-2, -1)) * mod.qk_scale.exp()[None, :, None, None]
    else:
        s = q @ k.transpose(-2, -1) / math.sqrt(mod.dh)
    if out[1] is not None:
        calls.append((out[1].float().cpu(), s.float().cpu()))
h = model.cross_attn.register_forward_hook(hook)
mass = defaultdict(float); cnt = defaultdict(int); score = defaultdict(float)
with torch.no_grad():
    eps = episodes(1234, 64)
    chunk = [s for _, s in eps]
    tel = model(collate(chunk, r.device), n_iters=8, collect=True).telemetry
    Bm = tel["bind_matrix"].float().cpu()
h.remove()
iters = calls[-8:]           # the 8 core iterations (the first call is the warm init read)
H_ = iters[0][0].shape[1]
for j, s in enumerate(chunk):
    toks = s.tokens; q = toks.index("Q"); kind, span = row_types(toks)
    broken = {v for pair in s.meta["split_pairs"] for v in pair}
    for pair in range(8):
        v = f"X{2 * pair}"
        owner = int(Bm[j, 2 * pair].argmax())
        pos = {kind[k]: k for k in range(q) if toks[k] == v}
        for t, (probs, sc) in enumerate(iters):
            for hd in range(H_):
                for rt in ("sum", "prod", "hint"):
                    a0, b0 = span[pos[rt]]
                    mass[(t, hd, rt)] += float(probs[j, hd, owner, a0:b0].sum()) / (b0 - a0)
                    score[(t, hd, rt)] += float(sc[j, hd, owner, pos[rt]])
                    cnt[(t, hd, rt)] += 1
print(f"[{run}] owning slot, attention mass per token on its OWN rows (uniform ~0.0056), per iteration x head")
for t in range(8):
    print(f"  iter {t}: " + " | ".join(
        f"h{hd} sum {mass[(t, hd, 'sum')] / cnt[(t, hd, 'sum')]:.3f} prod {mass[(t, hd, 'prod')] / cnt[(t, hd, 'prod')]:.3f} "
        f"hint {mass[(t, hd, 'hint')] / cnt[(t, hd, 'hint')]:.3f}" for hd in range(H_)))
print("\nraw score the owning slot gives its variable's token in each row (iter 7):")
for hd in range(H_):
    print(f"  h{hd}: sum-row X {score[(7, hd, 'sum')] / cnt[(7, hd, 'sum')]:+.2f}  prod-row X {score[(7, hd, 'prod')] / cnt[(7, hd, 'prod')]:+.2f}  "
          f"hint-row X {score[(7, hd, 'hint')] / cnt[(7, hd, 'hint')]:+.2f}")
