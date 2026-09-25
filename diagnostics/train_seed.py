"""One seed of one arm on the 1,000-update single-stage contract (raw-weight eval
afterwards).  Arms: capped (bind_norm=capped x32), uncapped (softmax), transformer
(transformer_bind_match, untuned, parameter-matched).  Usage: train_seed.py ARM SEED"""
import os, sys, time
from dataclasses import replace
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import repro_1000 as r
from far2.config import TrainConfig
from far2.models import build_model
from far2.presets import get_preset
from far2.train import Stage, Trainer
from far2.utils import set_seed
arm, seed = sys.argv[1], int(sys.argv[2])
far2 = replace(get_preset("far2_bind"), n_iters_train=8, init_slots=14, workspace_init="input")
cfg = {"capped": replace(far2, bind_norm="capped", bind_sweeps=32),
       "uncapped": far2,
       "transformer": get_preset("transformer_bind_match"),
       # encoder-quality arms, all on the capped substrate
       "enc2": replace(far2, bind_norm="capped", bind_sweeps=32, n_encoder_layers=2),
       "rope": replace(far2, bind_norm="capped", bind_sweeps=32, enc_pos="rope"),
       "roles": replace(far2, bind_norm="capped", bind_sweeps=32, enc_clause_roles=True),
       "localqk": replace(far2, bind_norm="capped", bind_sweeps=32, enc_local_mix=True, enc_qk_norm=True),
       "route": replace(far2, bind_norm="capped", bind_sweeps=32, enc_local_mix=True, enc_qk_norm=True,
                        slot_query_entities=True),
       "clause": replace(far2, bind_norm="capped", bind_sweeps=32, enc_local_mix=True, enc_qk_norm=True,
                         slot_query_entities=True, slot_read_clauses=True),
       # upper bound: SPLIT forced by the task condition on each broken pair
       "route_oracle_split": replace(far2, init_slots=8, bind_norm="capped", bind_sweeps=32,
                                     enc_local_mix=True, enc_qk_norm=True, slot_query_entities=True,
                                     repr_policy="oracle", oracle_action="split"),
       "bm_identity": replace(far2, bind_norm="capped", bind_sweeps=32, enc_local_mix=True, enc_qk_norm=True,
                              slot_query_entities=True, bind_memory=True, bind_memory_roles="identity"),
       "bm_structural": replace(far2, bind_norm="capped", bind_sweeps=32, enc_local_mix=True, enc_qk_norm=True,
                                slot_query_entities=True, bind_memory=True, bind_memory_roles="structural")}[arm]
out = os.path.join(r.RUNS, f"seeds/{arm}_seed{seed}"); os.makedirs(out, exist_ok=True)
tc = TrainConfig(steps=4000, batch_size=128, device="mps", seed=seed, out_dir=out, warmup=266, ema=0.999,
                 ipt_alpha=0.0, cycle_supervision="final", checkpoint_steps=r.POINTS, log_every=50)
set_seed(seed)
unlock = set(os.environ.get("FAR2_UNLOCK", "").split(",")) - {""}
stage = Stage("only", 1.0, r.fc._mix(), unlocked=unlock, depth="fixed", bind_weight=0.0)
tr = Trainer(build_model(cfg), cfg, tc, [stage])
inner, calls = tr.opt.step, [0]
def counted(*a, **k):
    if calls[0] == r.STOP:
        raise r.Stop
    calls[0] += 1
    return inner(*a, **k)
tr.opt.step = counted
t0 = time.perf_counter()
try:
    tr.train()
except r.Stop:
    pass
print(f"trained {arm} seed {seed}: {calls[0]} updates in {(time.perf_counter() - t0) / 60:.1f} min "
      f"({tr.model.param_count():,} params)", flush=True)
