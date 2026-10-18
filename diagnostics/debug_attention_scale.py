"""Why is attention flat?  Compare the trained checkpoint with a fresh model of
the same config: std of the pre-softmax scores q.k/sqrt(dh) (per head), the
normalized entropy of the softmax, and the norms of Wq/Wk, for the encoder
self-attention and the slot cross-attention (first iteration)."""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
run = sys.argv[1] if len(sys.argv) > 1 else "run1000_capped32"
sys.argv = [sys.argv[0], run, "1000"]
src = open(os.path.join(HERE, "debug_encoder.py")).read().split("eps_tr, eps_te = episodes(777, 600)")[0]
ns = {"__file__": os.path.join(HERE, "debug_encoder.py")}
exec(compile(src, "debug_encoder_head", "exec"), ns)
torch, r, trained, collate, episodes = (ns[k] for k in ("torch", "r", "model", "collate", "episodes")); cfg = trained.cfg
from far2.models import build_model
from far2.utils import set_seed
set_seed(0)
fresh = build_model(cfg).to(r.device).eval()

def scores(attn, q_in, kv_in, pad):
    B, Lq, _ = q_in.shape; Lk = kv_in.shape[1]
    q = attn.wq(q_in).view(B, Lq, attn.h, attn.dh).transpose(1, 2)
    k = attn.wk(kv_in).view(B, Lk, attn.h, attn.dh).transpose(1, 2)
    s = (q @ k.transpose(-2, -1) / math.sqrt(attn.dh)).float()
    valid = pad[:, None, None, :].expand_as(s)
    std = torch.stack([s[:, h][valid[:, h]].std() for h in range(attn.h)])
    p = torch.softmax(s.masked_fill(~valid, -1e4), -1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1) / torch.log(pad.sum(-1).float())[:, None, None]
    return std.cpu(), float(ent.mean())

with torch.no_grad():
    eps = episodes(1234, 22)
    batch = collate([s for _, s in eps], r.device)
    pad = batch["pad_mask"]
    for name, m in (("fresh init", fresh), ("trained 1k", trained)):
        ids = batch["input_ids"]; L = ids.shape[1]
        x = m.tok(ids) + m.pos(torch.arange(L, device=ids.device))[None] + m.sig(batch["digit_sig"])
        blk = m.encoder[0]; h = blk.ln1(x)
        es, ee = scores(blk.attn, h, h, pad)
        H, _ = m._encode(batch)
        active = torch.zeros(ids.shape[0], cfg.max_slots, device=ids.device)
        active[:, :cfg.init_slots + cfg.n_registers] = 1
        Z0 = m._init_workspace(H, torch.where(pad, 0.0, -1e4)[:, None, None, :], active)
        cs, ce = scores(m.cross_attn, m.ln_cross(Z0), H, pad)
        print(f"{name:11s} encoder self-attn: score std/head {[round(float(v), 3) for v in es]} entropy {ee:.3f} "
              f"| |Wq| {float(blk.attn.wq.weight.norm()):.2f} |Wk| {float(blk.attn.wk.weight.norm()):.2f}")
        print(f"{'':11s} slot cross-attn:   score std/head {[round(float(v), 3) for v in cs]} entropy {ce:.3f} "
              f"| |Wq| {float(m.cross_attn.wq.weight.norm()):.2f} |Wk| {float(m.cross_attn.wk.weight.norm()):.2f}")
    print(f"\ntraining settings: lr 3e-4, grad clip 0.5, AdamW(0.9,0.95), wd 0.01; 1,000 updates")
