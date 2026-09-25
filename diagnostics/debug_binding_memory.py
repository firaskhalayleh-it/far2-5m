"""Inspect a trained binding memory: key geometry, memory norm per write,
read fidelity (does key_v @ M return variable v's own last filler?), and how
much of a read comes from OTHER variables' fillers (cross-talk)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import repro_1000 as r
import torch
import torch.nn.functional as F
from far2.config import ModelConfig
from far2.data import GENERATORS, collate
from far2.models import build_model
import random
run = sys.argv[1]; point = sys.argv[2] if len(sys.argv) > 2 else "1000"
ck = torch.load(os.path.join(r.RUNS, run, f"ckpt_step{point}.pt"), map_location="cpu", weights_only=False)
m = build_model(ModelConfig(**ck["model_cfg"])); m.load_state_dict(ck["model"], strict=False)
m.to(r.device).eval(); m.set_unlocked(set())
writes = []
orig = m._bm_write
def spy(M, Z, B_mat, role, key, present):
    own = B_mat @ m.bm_ln(Z)
    fill = m.bm_fill(torch.cat([own, role], dim=-1))
    out = orig(M, Z, B_mat, role, key, present)
    writes.append((fill.detach(), out.detach(), key.detach(), present))
    return out
m._bm_write = spy
rng = random.Random("1234:bm")
batch = collate([GENERATORS["symbreak"].sample(rng, 2) for _ in range(64)], r.device)
with torch.no_grad():
    m(batch, n_iters=8)
fill, M, key, present = writes[-1]
G = key[0] @ key[0].T
off = G[~torch.eye(16, dtype=torch.bool, device=G.device)]
print(f"[{run} @ {point}] rate sigmoid={float(torch.sigmoid(m.bm_rate)):.3f}")
print(f"key cosine off-diagonal: mean |cos| {float(off.abs().mean()):.3f}, max {float(off.abs().max()):.3f};"
      f" X4-X5 {float(G[4, 5]):.3f}")
print("memory norm after each write: " + " ".join(f"{float(w[1].norm(dim=(1, 2)).mean()):.1f}" for w in writes))
read = key @ M                                                        # [B,V,d]
fid = F.cosine_similarity(read, fill, dim=-1)[present.bool()]
print(f"read vs own last filler: mean cos {float(fid.mean()):.3f}; read norm {float(read.norm(dim=-1).mean()):.2f}"
      f" / filler norm {float(fill.norm(dim=-1).mean()):.2f}")
fn = F.normalize(fill, dim=-1)
sim = torch.einsum("bvd,bwd->bvw", F.normalize(read, dim=-1), fn)
eye = torch.eye(16, dtype=torch.bool, device=sim.device)[None].expand_as(sim)
print(f"read cos to own filler {float(sim[eye].mean()):.3f} vs to other variables' fillers {float(sim[~eye].mean()):.3f}")
print(f"fillers of X4 vs X5 (same slot): cos {float(F.cosine_similarity(fill[:, 4], fill[:, 5], dim=-1).mean()):.3f};"
      f" across pairs X4 vs X6: {float(F.cosine_similarity(fill[:, 4], fill[:, 6], dim=-1).mean()):.3f}")
