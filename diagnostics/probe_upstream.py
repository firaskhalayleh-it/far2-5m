"""Validate the ordering-bit probe and locate where the bit is lost.
Positive control: 'pair sum >= 10' from the owner slot (the model uses it: symmetric
answers are 95% right).  Upstream: the ordering bit from the entity states E
(first var, second var, and their difference), T=8, raw weights."""
import os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "run1000_capped32", "1000"]
import importlib.util
spec = importlib.util.spec_from_file_location("ds", os.path.join(HERE, "diagnose_selection.py"))
src = open(os.path.join(HERE, "diagnose_selection.py")).read().split("test_eps = episodes(1234, 256)")[0]
ns = {"__file__": os.path.join(HERE, "diagnose_selection.py")}
exec(compile(src, "diagnose_selection_head", "exec"), ns)
torch, r, model, collate, episodes, probe = ns["torch"], ns["r"], ns["model"], ns["collate"], ns["episodes"], ns["probe"]

@torch.no_grad()
def rows(eps, T=8):
    out_rows = []
    for b in range(0, len(eps), 64):
        chunk = [s for _, s in eps[b:b + 64]]
        tel = model(collate(chunk, r.device), n_iters=T, collect=True).telemetry
        Z, Bm, E = (tel[k].float().cpu() for k in ("Z_final", "bind_matrix", "entity_states"))
        for j, s in enumerate(chunk):
            vals = s.meta["vars"]
            for a_var, b_var in s.meta["split_pairs"]:
                ia, ib = int(a_var[1:]), int(b_var[1:])
                owner = int(Bm[j, ia].argmax())
                out_rows.append({"owner": Z[j, owner], "Ea": E[j, ia], "Eb": E[j, ib],
                                 "Ediff": E[j, ia] - E[j, ib],
                                 "order": float(vals[a_var] > vals[b_var]),
                                 "sum10": float(vals[a_var] + vals[b_var] >= 10)})
    return out_rows

tr, te = rows(episodes(777, 1200)), rows(episodes(1234, 256))
def p(feat, label, **kw):
    return probe([(x[feat], x[label]) for x in tr], [(x[feat], x[label]) for x in te], **kw)
maj = lambda label: max(sum(x[label] for x in te), len(te) - sum(x[label] for x in te)) / len(te)
print(f"broken pairs: train {len(tr)}, test {len(te)}")
print(f"POSITIVE CONTROL  'sum >= 10' from owner slot: {p('owner', 'sum10'):5.1%}  (majority {maj('sum10'):5.1%})")
print(f"ordering bit from owner slot (T=8):            {p('owner', 'order'):5.1%}  (majority {maj('order'):5.1%})")
print(f"ordering bit from entity state E_first:        {p('Ea', 'order'):5.1%}")
print(f"ordering bit from entity state E_second:       {p('Eb', 'order'):5.1%}")
print(f"ordering bit from E_first - E_second:          {p('Ediff', 'order'):5.1%}")
print(f"'sum >= 10' from E_first (upstream sanity):    {p('Ea', 'sum10'):5.1%}")
