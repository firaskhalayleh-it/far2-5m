"""Write/refresh the manifest that ties a run to the exact code that produced it.

    run_manifest.py start RUN_DIR -- CMD...   record commit, dirty state, command, device
    run_manifest.py finish RUN_DIR LOG        add duration and the parsed metrics

The manifest lives beside the checkpoints (gitignored) and a copy with the log
goes to diagnostics/results/ so it can be committed."""
import json, os, platform, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH_PATHS = ["far2", "scripts", "diagnostics", "tests"]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def research_dirty():
    """Tracked changes anywhere in the research paths, plus untracked files in
    far2/ and diagnostics/ (scripts/ and tests/ also hold shelved-track files
    that are deliberately untracked)."""
    tracked = git("status", "--porcelain", "--untracked-files=no", "--", *RESEARCH_PATHS)
    untracked = git("status", "--porcelain", "--", "far2", "diagnostics")
    return "\n".join(x for x in (tracked, untracked) if x)


def parse_metrics(text):
    m = {}
    t8 = re.search(r"^\s+8 \|\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", text, re.M)
    if t8:
        m["exact_T8"], m["symmetric_T8"], m["broken_T8"] = (float(x) for x in t8.groups())
    curve = re.findall(r"^\s+(\d+|None) \|\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", text, re.M)
    if curve:
        m["curve"] = {T: {"exact": float(e), "symmetric": float(s), "broken": float(b)} for T, e, s, b in curve}
    for key, pat in {"correct": r"correct\s+([\d.]+)%", "partner": r"partner \(selection\)\s+([\d.]+)%",
                     "half_sum": r"half-sum \(treated symmetric\)\s+([\d.]+)%",
                     "other_pair": r"other pair \(binding\)\s+([\d.]+)%", "arithmetic": r"none \(arithmetic\)\s+([\d.]+)%",
                     "selection": r"selection correct/\(correct\+partner\):\s+([\d.]+)%",
                     "probe_slot_order": r"ordering bit from owner slot \(T=8\):\s+([\d.]+)%",
                     "probe_sum10": r"'sum >= 10' from owner slot:\s+([\d.]+)%",
                     "probe_hint_left": r"own hint-row token -> 'left of GEQ':\s+([\d.]+)%"}.items():
        hit = re.search(pat, text)
        if hit:
            m[key] = float(hit.group(1))
    return m


if __name__ == "__main__":
    if sys.argv[1] == "dirty":
        print(research_dirty())
        sys.exit(0)
    mode, run_dir = sys.argv[1], sys.argv[2]
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run_manifest.json")
    if mode == "start":
        cmd = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        dirty = research_dirty()
        man = {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
               "research_tree_dirty": bool(dirty), "dirty_files": dirty.splitlines(),
               "command": cmd, "env": {k: v for k, v in os.environ.items() if k.startswith("FAR2_")},
               "host": platform.platform(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "started_epoch": time.time()}
        json.dump(man, open(path, "w"), indent=2)
    else:
        man = json.load(open(path))
        text = open(sys.argv[3]).read()
        man["duration_min"] = round((time.time() - man.pop("started_epoch")) / 60, 1)
        man["metrics"] = parse_metrics(text)
        json.dump(man, open(path, "w"), indent=2)
        print(json.dumps({k: man[k] for k in ("commit", "research_tree_dirty", "duration_min")}))
