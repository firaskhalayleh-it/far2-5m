"""P4 control: the answer must come from the workspace, not from the roles.
Evaluate with every memory write fed another episode's slots (batch roll), so
roles and keys are intact but the content is wrong.  Broken and symmetric
accuracy should fall to chance-level."""
import os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import repro_1000 as r
import torch
from far2.config import ModelConfig
from far2.models import build_model
run = sys.argv[1]
ck = torch.load(os.path.join(r.RUNS, run, "ckpt_step1000.pt"), map_location="cpu", weights_only=False)
m = build_model(ModelConfig(**ck["model_cfg"])); m.load_state_dict(ck["model"], strict=False)
m.to(r.device).eval(); m.set_unlocked(set())
tok, ex, sym, brk, _ = r.score(m, 8)
print(f"[{run}] intact workspace:            exact {ex:5.1%} symmetric {sym:5.1%} broken {brk:5.1%}")
orig = m._bm_write
m._bm_write = types.MethodType(lambda self, M, Z, B_mat, role, key, present:
                               orig(M, torch.roll(Z, 1, dims=0), torch.roll(B_mat, 1, dims=0), role, key, present), m)
tok, ex, sym, brk, _ = r.score(m, 8)
print(f"[{run}] workspace from another episode: exact {ex:5.1%} symmetric {sym:5.1%} broken {brk:5.1%}")
