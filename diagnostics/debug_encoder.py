"""Debug the encoder on a trained checkpoint (eval only, no training).

A. Stage probes at the variable's own hint-row token for "left of GEQ":
   input embedding / after self-attention residual / after MLP / after final LN.
B. Encoder self-attention at hint-row variable tokens: attention mass on the
   left neighbour, right neighbour, itself, rest of its clause, other
   occurrences of the same variable, and everything else (per head), compared
   with the same variable's sum-row token.
C. Can attention reach the neighbour at all?  q.k score of token k to k+1 vs
   to other tokens, split by position (early vs late in the sequence).
D. Embedding scale: token vs position vs digit-significance norms; how much
   late positions moved from initialization.
E. Gradient reaching the encoder output H by row type (does anything
   downstream ask for hint-row information?).
F. Slot cross-attention (final iteration) mass per token by row type.
"""

import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
run = sys.argv[1] if len(sys.argv) > 1 else "run1000_capped32"
sys.argv = [sys.argv[0], run, sys.argv[2] if len(sys.argv) > 2 else "1000"]
src = open(os.path.join(HERE, "diagnose_selection.py")).read().split("test_eps = episodes(1234, 256)")[0]
ns = {"__file__": os.path.join(HERE, "diagnose_selection.py")}
exec(compile(src, "diagnose_selection_head", "exec"), ns)
torch, r, model, collate, episodes, probe = (ns[k] for k in ("torch", "r", "model", "collate", "episodes", "probe"))
F = torch.nn.functional
from far2.models.layers import MASK_FILL  # noqa: E402


def row_types(toks):
    """Per position: (row type, clause start, clause end).  Row types:
    hint (X GEQ X), sum (X PLUS X ...), prod (X TIMES X ...), filler (D TIMES D ...), other."""
    q = toks.index("Q")
    out, start = [], 0
    bounds = []
    for k, t in enumerate(toks):
        if t in ("BOS", "CTX", "SEP", "Q") and k > 0:
            bounds.append((start, k))
            start = k
    bounds.append((start, len(toks)))
    kind = [None] * len(toks)
    span = [None] * len(toks)
    for a, b in bounds:
        body = toks[a + 1:b] if toks[a] in ("SEP", "CTX") else toks[a:b]
        if a >= q:
            rt = "query"
        elif "GEQ" in body:
            rt = "hint"
        elif "PLUS" in body:
            rt = "sum"
        elif "TIMES" in body and body[0].startswith("X"):
            rt = "prod"
        elif "TIMES" in body:
            rt = "filler"
        else:
            rt = "other"
        for k in range(a, b):
            kind[k], span[k] = rt, (a, b)
    return kind, span


@torch.no_grad()
def encoder_internals(batch):
    """Re-run the encoder by hand, keeping every stage and attention map."""
    ids = batch["input_ids"]
    L = ids.shape[1]
    x = model.tok(ids) + model.pos(torch.arange(L, device=ids.device))[None]
    if model.sig is not None:
        x = x + model.sig(batch["digit_sig"])
    mask = torch.where(batch["pad_mask"], 0.0, MASK_FILL)[:, None, None, :]
    stages = {"input": x}
    probs_all = []
    for i, blk in enumerate(model.encoder):
        h = blk.ln1(x)
        a, probs = blk.attn(h, h, mask, need_probs=True)
        x = x + a
        stages[f"L{i}_after_attn"] = x
        x = x + blk.mlp(blk.ln2(x))
        stages[f"L{i}_after_mlp"] = x
        probs_all.append(probs)
    stages["H"] = model.enc_ln(x)
    return stages, probs_all


eps_tr, eps_te = episodes(777, 600), episodes(1234, 256)

# ---------------- A + B ------------------------------------------------------
def collect(eps, want_attn):
    rows, attn = [], defaultdict(lambda: defaultdict(list))
    for b in range(0, len(eps), 64):
        chunk = [s for _, s in eps[b:b + 64]]
        batch = collate(chunk, r.device)
        stages, probs_all = encoder_internals(batch)
        stages = {k: v.float().cpu() for k, v in stages.items()}
        probs = probs_all[0].float().cpu() if want_attn else None      # [B,h,L,L]
        for j, s in enumerate(chunk):
            toks = s.tokens
            kind, span = row_types(toks)
            q = toks.index("Q")
            for a_var, b_var in s.meta["split_pairs"]:
                for v in (a_var, b_var):
                    pos = [k for k in range(q) if toks[k] == v]
                    hint = [k for k in pos if kind[k] == "hint"][0]
                    summ = [k for k in pos if kind[k] == "sum"][0]
                    left = float(toks[hint + 1] == "GEQ")
                    rows.append({**{st: stages[st][j, hint] for st in stages}, "left": left,
                                 "position": hint})
                    if want_attn:
                        for label, k in (("hint", hint), ("sum", summ)):
                            p = probs[j, :, k, :len(toks)]                   # [h, L]
                            a0, b0 = span[k]
                            same = [m for m in pos if m != k]
                            clause = [m for m in range(a0, b0) if m not in (k - 1, k, k + 1)]
                            parts = {"left nbr": p[:, k - 1], "self": p[:, k],
                                     "right nbr": p[:, k + 1],
                                     "rest of clause": p[:, clause].sum(-1),
                                     "same var elsewhere": p[:, same].sum(-1)}
                            parts["everything else"] = 1 - sum(parts.values())
                            for name, val in parts.items():
                                attn[label][name].append(val)
    return rows, attn


tr, _ = collect(eps_tr, False)
te, attn = collect(eps_te, True)
print(f"[{run}] encoder debug, broken-pair variables, train {len(tr)} test {len(te)}\n")
print("A. probe 'left of GEQ' at the variable's own hint-row token, by encoder stage:")
for st in [k for k in tr[0] if k not in ("left", "position")]:
    acc = probe([(x[st], x["left"]) for x in tr], [(x[st], x["left"]) for x in te])
    print(f"   {st:16s} {acc:6.1%}")

print("\nB. encoder self-attention from the variable token (mean over tokens), per head:")
for label in ("hint", "sum"):
    print(f"   from its {label}-row token:")
    for name, vals in attn[label].items():
        v = torch.stack(vals).mean(0)
        print(f"     {name:20s} " + "  ".join(f"h{i}:{float(x):5.1%}" for i, x in enumerate(v)))

# ---------------- C: can a head score the right neighbour highly? -----------
print("\nC. rank of the right neighbour among all keys for a hint-row variable token (lower = easier to find),")
print("   using each head's own q.k scores, by position of the token:")
with torch.no_grad():
    blk = model.encoder[0]
    ranks = defaultdict(lambda: defaultdict(list))
    for b in range(0, 256, 64):
        chunk = [s for _, s in eps_te[b:b + 64]]
        batch = collate(chunk, r.device)
        stages, _ = encoder_internals(batch)
        h = blk.ln1(stages["input"])
        B_, L, d = h.shape
        H_, dh = blk.attn.h, blk.attn.dh
        qv = blk.attn.wq(h).view(B_, L, H_, dh).transpose(1, 2)
        kv = blk.attn.wk(h).view(B_, L, H_, dh).transpose(1, 2)
        sc = (qv @ kv.transpose(-2, -1)).float().cpu()                     # [B,h,L,L]
        for j, s in enumerate(chunk):
            toks = s.tokens
            kind, _ = row_types(toks)
            n = len(toks)
            for k in range(1, toks.index("Q")):
                if kind[k] == "hint" and toks[k].startswith("X"):
                    bucket = "pos<100" if k < 100 else "pos>=100"
                    for hd in range(H_):
                        row = sc[j, hd, k, :n]
                        ranks[bucket][hd].append(int((row > row[k + 1]).sum()))
    for bucket in ("pos<100", "pos>=100"):
        if ranks[bucket]:
            print(f"   {bucket:9s} median rank of right neighbour: " +
                  "  ".join(f"h{hd}:{sorted(v)[len(v) // 2]:3d}" for hd, v in sorted(ranks[bucket].items())) +
                  f"   (n={len(ranks[bucket][0])}, sequence length ~{n})")

# ---------------- D: embedding scale --------------------------------------------
init = torch.nn.Embedding(model.pos.num_embeddings, model.pos.embedding_dim)  # default init reference
tokw, posw = model.tok.weight.float().cpu(), model.pos.weight.float().cpu()
print("\nD. embedding norms: token {:.1f} | position {:.1f} | digit significance {:.2f} | "
      "default-init reference {:.1f}".format(
          tokw.norm(dim=-1).mean(), posw.norm(dim=-1).mean(),
          model.sig.weight.float().norm(dim=-1).mean() if model.sig is not None else 0,
          init.weight.norm(dim=-1).mean()))
lens = [len(s.tokens) for _, s in eps_te]
print(f"   symbreak sequence lengths: {min(lens)}-{max(lens)}; hint rows occupy the last ~25 context tokens")
adj = F.cosine_similarity(posw[:-1], posw[1:], dim=-1)
print("   cosine(pos k, pos k+1): positions 0-50 {:.3f} | 100-150 {:.3f} | 150-200 {:.3f} | 300-400 {:.3f}".format(
    adj[0:50].mean(), adj[100:150].mean(), adj[150:200].mean(), adj[300:400].mean()))

# ---------------- E + F: gradient and slot attention by row type ------------
print("\nE/F. per-token gradient norm of the answer loss at H, and final slot cross-attention mass per token:")
grads, xattn, counts = defaultdict(float), defaultdict(float), defaultdict(int)
captured = {}
def _keep(m, i, o):
    o.retain_grad()
    captured["H"] = o
hook = model.enc_ln.register_forward_hook(_keep)
for b in range(0, 256, 64):
    chunk = [s for _, s in eps_te[b:b + 64]]
    batch = collate(chunk, r.device)
    model.zero_grad(set_to_none=True)
    out = model(batch, n_iters=8, collect=True)
    V = out.logits.shape[-1]
    F.cross_entropy(out.logits.reshape(-1, V), batch["target_ids"].reshape(-1),
                    ignore_index=-100).backward()
    g = captured["H"].grad.float().norm(dim=-1).cpu()                        # [B,L]
    ca = out.telemetry["cross_attn"].float().cpu()                           # [B,S,L]
    act = out.telemetry["active_mask"].float().cpu()
    for j, s in enumerate(chunk):
        kind, _ = row_types(s.tokens)
        mass = (ca[j] * act[j][:, None]).sum(0) / act[j].sum()               # [L]
        for k, rt in enumerate(kind):
            grads[rt] += float(g[j, k])
            xattn[rt] += float(mass[k])
            counts[rt] += 1
hook.remove()
base = grads["sum"] / counts["sum"]
for rt in ("sum", "prod", "hint", "filler", "query", "other"):
    if counts[rt]:
        print(f"   {rt:7s} grad/token {grads[rt] / counts[rt]:.2e} ({grads[rt] / counts[rt] / base:5.2f}x sum)"
              f"   slot attention/token {xattn[rt] / counts[rt]:.4f}")
