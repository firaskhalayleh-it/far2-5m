"""Render runs/ into the research dashboard: baseline vs FAR², difficulty →
compute allocation, ambiguity → representation allocation, wrong trajectory →
reconsideration, ablation evidence. Output: a single self-contained HTML."""

import argparse
import base64
import io
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from far2.utils.logging import read_jsonl


def find_runs(root: str) -> dict[str, dict]:
    runs = {}
    for dirpath, _, files in os.walk(root):
        if "config.json" not in files:
            continue
        name = os.path.basename(dirpath)
        run = {"dir": dirpath,
               "config": json.load(open(os.path.join(dirpath, "config.json"))),
               "log": [], "evals": []}
        lp = os.path.join(dirpath, "train_log.jsonl")
        if os.path.exists(lp):
            run["log"] = read_jsonl(lp)
        ed = os.path.join(dirpath, "eval")
        if os.path.isdir(ed):
            for f in sorted(os.listdir(ed)):
                if f.endswith(".jsonl"):
                    run["evals"] += read_jsonl(os.path.join(ed, f))
        runs[name] = run
    return runs


def fig_to_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


def img(title: str, b64: str) -> str:
    return (f"<section><h2>{title}</h2>"
            f'<img alt="{title}" src="data:image/png;base64,{b64}"></section>')


def plot_losses(runs) -> str:
    plt.figure(figsize=(8, 4.2))
    for name, run in runs.items():
        if run["log"]:
            plt.plot([r["step"] for r in run["log"]],
                     [r["loss"] for r in run["log"]], label=name, lw=1.2)
    plt.xlabel("step"); plt.ylabel("loss"); plt.yscale("log")
    plt.legend(fontsize=7); plt.grid(alpha=.3)
    return img("Training loss", fig_to_b64())


def plot_acc_vs_depth(runs) -> str:
    plotted = False
    plt.figure(figsize=(8, 4.2))
    for name, run in runs.items():
        by_t = defaultdict(list)
        for r in run["evals"]:
            if isinstance(r.get("n_iters"), int):
                by_t[r["n_iters"]].append(r["correct"])
        if len(by_t) >= 3:
            ts = sorted(by_t)
            plt.plot(ts, [sum(by_t[t]) / len(by_t[t]) for t in ts],
                     marker="o", label=name)
            plotted = True
    if not plotted:
        plt.close()
        return ""
    plt.xlabel("reasoning iterations T"); plt.ylabel("accuracy")
    plt.xscale("log", base=2); plt.ylim(0, 1); plt.legend(fontsize=7); plt.grid(alpha=.3)
    return img("Accuracy = f(reasoning iterations)", fig_to_b64())


def plot_family_acc(runs) -> str:
    fams = sorted({r["family"] for run in runs.values() for r in run["evals"]})
    if not fams:
        return ""
    plt.figure(figsize=(9, 4.2))
    width = 0.8 / max(len(runs), 1)
    for i, (name, run) in enumerate(runs.items()):
        accs = []
        for f in fams:
            sel = [r["correct"] for r in run["evals"] if r["family"] == f]
            accs.append(sum(sel) / len(sel) if sel else 0)
        plt.bar([x + i * width for x in range(len(fams))], accs, width, label=name)
    plt.xticks([x + 0.4 for x in range(len(fams))], fams, rotation=30, fontsize=8)
    plt.ylabel("accuracy"); plt.ylim(0, 1); plt.legend(fontsize=7); plt.grid(axis="y", alpha=.3)
    return img("Baseline vs FAR² per family", fig_to_b64())


def plot_compute_alloc(runs) -> str:
    plotted = False
    plt.figure(figsize=(8, 4.2))
    for name, run in runs.items():
        by_d = defaultdict(list)
        for r in run["evals"]:
            if "iters_used" in r:
                by_d[r["difficulty"]].append(r["iters_used"])
        if len(by_d) >= 3:
            ds = sorted(by_d)
            plt.plot(ds, [sum(by_d[d]) / len(by_d[d]) for d in ds],
                     marker="s", label=name)
            plotted = True
    if not plotted:
        plt.close()
        return ""
    plt.xlabel("difficulty"); plt.ylabel("iterations used")
    plt.legend(fontsize=7); plt.grid(alpha=.3)
    return img("Difficulty → compute allocation", fig_to_b64())


def plot_repr_alloc(runs) -> str:
    labels, colls, ctrls = [], [], []
    for name, run in runs.items():
        c = [r["active_slots"] for r in run["evals"]
             if r.get("requires_distinction") == 1 and "active_slots" in r]
        n = [r["active_slots"] for r in run["evals"]
             if r.get("requires_distinction") == 0 and "active_slots" in r]
        if c and n:
            labels.append(name)
            colls.append(sum(c) / len(c))
            ctrls.append(sum(n) / len(n))
    if not labels:
        return ""
    plt.figure(figsize=(7, 4))
    x = range(len(labels))
    plt.bar([i - 0.18 for i in x], colls, 0.36, label="broken-pair query")
    plt.bar([i + 0.18 for i in x], ctrls, 0.36, label="symmetric-control query")
    plt.xticks(list(x), labels, rotation=20, fontsize=8)
    plt.ylabel("mean active slots"); plt.legend(fontsize=8); plt.grid(axis="y", alpha=.3)
    return img("Ambiguity → representation allocation", fig_to_b64())


def plot_recovery(runs) -> str:
    labels, vals = [], []
    for name, run in runs.items():
        hooked = [r for r in run["evals"]
                  if r.get("attractor_hit_at") and r.get("reconsider_label") == 1]
        if hooked:
            labels.append(name)
            vals.append(sum(r["correct"] for r in hooked) / len(hooked))
    if not labels:
        return ""
    plt.figure(figsize=(7, 3.6))
    plt.bar(labels, vals, color="#B45A08")
    plt.xticks(rotation=20, fontsize=8)
    plt.ylabel("recovery rate"); plt.ylim(0, 1); plt.grid(axis="y", alpha=.3)
    return img("Wrong trajectory → reconsideration (recovery)", fig_to_b64())


def ablation_table(root: str) -> str:
    path = os.path.join(root, "ablation_summary.json")
    if not os.path.exists(path):
        return ""
    rows = json.load(open(path))
    by = defaultdict(list)
    for r in rows:
        by[r["preset"]].append(r)
    html = ["<section><h2>Ablation evidence</h2><table><tr><th>config</th>"
            "<th>acc</th><th>acc/PFLOP</th><th>split P/R</th><th>recovery</th>"
            "<th>unsupported</th><th>goal viol.</th></tr>"]
    for preset, rs in by.items():
        def m(k, scale=1.0):
            vals = [r[k] * scale for r in rs if r[k] == r[k]]
            return sum(vals) / len(vals) if vals else float("nan")
        html.append(
            f"<tr><td>{preset}</td><td>{m('acc_overall'):.1%}</td>"
            f"<td>{m('acc_per_flop', 1e15):.2f}</td>"
            f"<td>{m('split_precision'):.2f}/{m('split_recall'):.2f}</td>"
            f"<td>{m('recovery_rate'):.1%}</td>"
            f"<td>{m('unsupported_assertion'):.1%}</td>"
            f"<td>{m('goal_violation'):.1%}</td></tr>")
    html.append("</table></section>")
    return "".join(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = find_runs(args.root)
    if not runs:
        print(f"no runs found under {args.root}")
        sys.exit(1)

    sections = [plot_losses(runs), plot_acc_vs_depth(runs), plot_family_acc(runs),
                plot_compute_alloc(runs), plot_repr_alloc(runs),
                plot_recovery(runs), ablation_table(args.root)]
    body = "\n".join(s for s in sections if s)
    html = ("<title>FAR²-5M results</title><style>body{font-family:system-ui;"
            "max-width:960px;margin:2rem auto;padding:0 1rem;background:#fff;color:#1b2331}"
            "img{max-width:100%}table{border-collapse:collapse}"
            "td,th{border:1px solid #ccc;padding:4px 10px;font-size:14px}"
            "h1{font-size:1.6rem}</style>"
            f"<h1>FAR²-5M — research dashboard</h1><p>{len(runs)} runs under "
            f"<code>{args.root}</code></p>{body}")
    out = args.out or os.path.join(args.root, "report.html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(html)
    print(f"report: {out}")


if __name__ == "__main__":
    main()
