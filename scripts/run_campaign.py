"""The evaluation campaign as a FUNNEL — training is expensive and selective,
evaluation is cheap and large, and only survivors earn promotion:

  smoke    (scripts/smoke_test.py, run separately)   catch code failures
  pilot    ~2.5-3k steps, seed 0, large evals        does the mechanism show
                                                     ANY signal?
  confirm  5-10k steps, thousands of eval episodes   only for pilot survivors
  seeds    3-5 seeds                                 only for important survivors
  grid     the 17-arm ablation matrix                only after core gates pass

The pilot runs the killer-result chain in order, with stop rules:

  P0 baselines -> P1 recurrence [STOP on fail] -> P2 depth extrapolation
  -> P3a oracle quartet on the binding substrate [STOP if the benchmark
     does not require representation] -> R0/R1/R4 probes
  -> P3b autonomous controller quartet -> R8 usage gates + R9 do()
  -> cross-domain transfer (held-out families)

Deferred until the core survives: P5a/P5b reconsideration+halting, P6
epistemics, P7 protected goal, the full ablation grid, extra seeds.
Gate FAILURES do not stop the chain unless marked — a clean negative is a
result; only the two structural stop rules abort downstream steps whose
premise died. Everything is logged under runs/campaign/."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


def _package_hash() -> str:
    """One hash over every .py in far2/ plus the shared script helpers — a
    step's behavior depends on the whole imported package, not just its own
    script file (review blocker: fingerprints covered script+args only)."""
    h = hashlib.sha256()
    files = [os.path.join(SCRIPTS, "_common.py")]
    for base, _, names in sorted(os.walk(os.path.join(ROOT, "far2"))):
        files += [os.path.join(base, n) for n in sorted(names)
                  if n.endswith(".py")]
    for f in files:
        h.update(f.encode())
        h.update(open(f, "rb").read())
    return h.hexdigest()


def fingerprint(st, pkg_hash: str) -> str:
    """Identity of a step: a completed step is only skippable on resume if
    the script bytes, its arguments, the far2 package code, AND the upstream
    checkpoints it consumes (--run/--control) are all unchanged."""
    h = hashlib.sha256()
    h.update(open(os.path.join(SCRIPTS, st["script"]), "rb").read())
    h.update(json.dumps(st["argv"]).encode())
    h.update(pkg_hash.encode())
    argv = st["argv"]
    for i, a in enumerate(argv[:-1]):
        if a in ("--run", "--control"):
            ck = os.path.join(ROOT, argv[i + 1], "ckpt.pt")
            h.update(ck.encode())
            if os.path.exists(ck):
                h.update(hashlib.sha256(open(ck, "rb").read()).hexdigest()
                         .encode())
            else:
                h.update(b"missing")
    return h.hexdigest()[:16]


def step(name, script, *argv, stop_on_fail=False, est_min=0):
    return {"name": name, "script": script, "argv": list(argv),
            "stop_on_fail": stop_on_fail, "est_min": est_min}


def pilot_chain(seed: int, out: str, mini: bool) -> list[dict]:
    s = str(seed)
    trn = ["--out", out] + (["--mini"] if mini else [])
    n_probe = "128" if mini else "512"
    n_interv = "120" if mini else "400"
    chain = [
        step("P0-transformer", "run_phase0.py", "--model", "transformer",
             "--steps", "3000", "--eval-samples", "300", "--seed", s, *trn,
             est_min=18),
        step("P0-recurrent", "run_phase0.py", "--model", "recurrent",
             "--steps", "3000", "--eval-samples", "300", "--seed", s, *trn,
             est_min=30),
        step("P0-slot-recurrent", "run_phase0.py", "--model", "slot_recurrent",
             "--steps", "3000", "--eval-samples", "300", "--seed", s, *trn,
             est_min=20),
        step("P1-recurrence", "run_phase1.py", "--steps", "2500",
             "--eval-samples", "300", "--seed", s, *trn,
             stop_on_fail=True, est_min=25),
        step("P2-extrapolation", "run_phase2.py", "--steps", "3000",
             "--eval-samples", "250", "--seed", s, *trn, est_min=45),
        step("P3a-oracle-quartet-bind", "run_phase3a.py", "--substrate",
             "bind", "--steps", "2500", "--eval-samples", "400", "--seed", s,
             *trn, stop_on_fail=True, est_min=170),
        step("R0R1R4-probes", "run_probes.py", "--strict",
             "--run", f"{out}/phase3a_oracle_split_bind_seed{seed}",
             "--control", f"{out}/phase3a_fixed16_bind_seed{seed}",
             "--n", n_probe, "--out", out, est_min=10),
        # P4 is a hard stop: with no validated signal there is nothing for
        # P3b to train on (and run_phase3b now refuses a silent default)
        step("P4-signal-tournament", "run_phase4.py", "--substrate", "bind",
             "--run", f"{out}/phase3a_oracle_split_bind_seed{seed}",
             "--seed", s, *trn, stop_on_fail=True, est_min=6),
        step("P3b-autonomous-bind", "run_phase3b.py", "--substrate", "bind",
             "--steps", "2500", "--eval-samples", "400", "--seed", s, *trn,
             est_min=95),
        step("R8R9-interventions", "run_interventions.py",
             "--run", f"{out}/phase3b_adaptive_bind_seed{seed}",
             "--ops", "split", "expand", "--n", n_interv, "--difficulty", "4",
             "--iters", "8", "--out", out, est_min=25),
        step("transfer", "run_transfer.py", "--eval-samples", "500",
             "--seed", s, *trn, est_min=20),
    ]
    return chain


def run_step(st, log_dir):
    log_path = os.path.join(log_dir, f"{st['name']}.log")
    cmd = [sys.executable, os.path.join(SCRIPTS, st["script"])] + st["argv"]
    t0 = time.time()
    with open(log_path, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=ROOT).returncode
    mins = (time.time() - t0) / 60
    text = open(log_path).read()
    crashed = "Traceback (most recent call last)" in text
    return rc, mins, log_path, text.splitlines()[-12:], crashed


def main():
    p = argparse.ArgumentParser(description="FAR2-5M staged campaign funnel")
    p.add_argument("--stage", choices=["pilot"], default="pilot",
                   help="confirm/seeds/grid are launched manually for "
                        "survivors — promotion is a human decision")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fresh", action="store_true",
                   help="ignore recorded progress and rerun every step")
    p.add_argument("--mini", action="store_true",
                   help="ISOLATED SMOKE of the whole chain in runs_mini/: "
                        "~10x smaller runs, gate failures expected and "
                        "ignored — only crashes count; stop rules off")
    args = p.parse_args()

    out = "runs_mini" if args.mini else "runs"
    chain = pilot_chain(args.seed, out, args.mini)
    tag = f"{args.stage}_mini" if args.mini else args.stage
    log_dir = os.path.join(ROOT, out, "campaign")
    os.makedirs(log_dir, exist_ok=True)
    state_path = os.path.join(log_dir, f"{tag}_state.json")
    state = {}
    if os.path.exists(state_path) and not args.fresh:
        state = json.load(open(state_path))

    est = sum(st["est_min"] for st in chain)
    mode = "MINI SMOKE (crash check only)" if args.mini else "real pilot"
    print(f"[campaign] {mode} · stage={args.stage} seed={args.seed} — "
          f"{len(chain)} steps"
          + ("" if args.mini else f", rough estimate {est/60:.1f} h"))
    print(f"[campaign] logs: {log_dir}/<step>.log · state: {state_path}\n",
          flush=True)

    stopped = None
    pkg_hash = _package_hash()
    for st in chain:
        rec = state.get(st["name"])
        fp = fingerprint(st, pkg_hash)
        if rec and rec.get("done") and rec.get("fingerprint") == fp:
            print(f"  {st['name']:<26} SKIP (already {rec['result']}, "
                  f"{rec['minutes']:.0f} min, fingerprint match)", flush=True)
            if rec["result"] == "FAIL" and st["stop_on_fail"] and not args.mini:
                stopped = st["name"]
                break
            continue
        if rec and rec.get("done"):
            print(f"  {st['name']:<26} STALE (code/config changed since the "
                  "recorded run) — rerunning", flush=True)
        print(f"  {st['name']:<26} running"
              + ("" if args.mini else f" (est ~{st['est_min']} min)") + " ...",
              flush=True)
        rc, mins, log_path, tail, crashed = run_step(st, log_dir)
        if args.mini:
            result = "CRASH" if crashed or (rc not in (0, 1)) else "OK"
        else:
            result = "PASS" if rc == 0 else ("CRASH" if crashed else "FAIL")
        state[st["name"]] = {"done": True, "result": result, "minutes": mins,
                             "log": log_path, "fingerprint": fp}
        json.dump(state, open(state_path, "w"), indent=2)
        print(f"  {st['name']:<26} {result} in {mins:.1f} min", flush=True)
        for line in tail[-6:]:
            print(f"      | {line}")
        if not args.mini and result != "PASS" and st["stop_on_fail"]:
            stopped = st["name"]
            break

    print("\n[campaign] funnel summary")
    bad = 0
    for st in chain:
        rec = state.get(st["name"])
        mark = rec["result"] if rec else ("BLOCKED" if stopped else "PENDING")
        bad += mark == "CRASH"
        print(f"  {st['name']:<26} {mark}")
    if args.mini:
        verdict = ("FAILED — fix crashes before the real pilot" if bad else
                   "CLEAN — every script ran end-to-end; gate FAILs above "
                   "are expected at smoke scale")
        print(f"\n[campaign] mini smoke {verdict}")
        sys.exit(1 if bad else 0)
    if stopped:
        print(f"\n[campaign] STOP RULE fired at {stopped} — downstream steps "
              "whose premise died were not run. Investigate before spending "
              "more compute (that is the funnel working, not a crash).")
        sys.exit(2)
    print("\n[campaign] pilot complete. Promotion to confirm/seeds/grid is a "
          "human decision — read the gate outputs first.")
    sys.exit(0)


if __name__ == "__main__":
    main()
