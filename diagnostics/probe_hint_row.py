"""Is the hint row encoded cleanly where it sits?  Linear probe on the encoder
state H at the variable's OWN token inside its hint row ("X4 GEQ X5"), target
'this token is left of GEQ' (= this variable holds the larger root).  Compared
with the same variable's tokens in its sum and product rows (which carry no
ordering locally) and the GEQ token itself.  Broken pairs only, raw weights."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "run1000_capped32", "1000"]
src = open(os.path.join(HERE, "diagnose_selection.py")).read().split("test_eps = episodes(1234, 256)")[0]
ns = {"__file__": os.path.join(HERE, "diagnose_selection.py")}
exec(compile(src, "diagnose_selection_head", "exec"), ns)
torch, r, model, collate, episodes, probe = (ns[k] for k in ("torch", "r", "model", "collate", "episodes", "probe"))

@torch.no_grad()
def rows(eps):
    out = []
    for b in range(0, len(eps), 64):
        chunk = [s for _, s in eps[b:b + 64]]
        batch = collate(chunk, r.device)
        H = model(batch, n_iters=0, collect=True).telemetry["H"].float().cpu()
        ids = batch["input_ids"].cpu()
        for j, s in enumerate(chunk):
            assert r.VOCAB.decode(ids[j, :len(s.tokens)].tolist()) == s.tokens
            toks = s.tokens
            for a_var, b_var in s.meta["split_pairs"]:
                for v in (a_var, b_var):
                    pos = [k for k, t in enumerate(toks[:toks.index("Q")]) if t == v]
                    hint = [k for k in pos if toks[k + 1] == "GEQ" or toks[k - 1] == "GEQ"]
                    other = [k for k in pos if k not in hint]
                    assert len(hint) == 1 and len(other) == 2
                    k = hint[0]
                    left = float(toks[k + 1] == "GEQ")
                    geq = k + 1 if left else k - 1
                    out.append({"hint_tok": H[j, k], "geq_tok": H[j, geq],
                                "row_toks": H[j, other].mean(0), "left": left,
                                "pair_bit": left if int(v[1:]) % 2 == 0 else 1.0 - left,
                                "first": float(int(v[1:]) % 2 == 0)})
    return out

tr, te = rows(episodes(777, 1200)), rows(episodes(1234, 256))
def p(feat, label="left"):
    return probe([(x[feat], x[label]) for x in tr], [(x[feat], x[label]) for x in te])
print(f"broken-pair variable occurrences: train {len(tr)}, test {len(te)}; majority 50%")
print(f"H at the variable's own hint-row token -> 'left of GEQ': {p('hint_tok'):5.1%}")
print(f"H at that row's GEQ token  -> 'pair's first var is left': {p('geq_tok', 'pair_bit'):5.1%}")
print(f"H at the hint-row var token -> 'pair's first var is left': {p('hint_tok', 'pair_bit'):5.1%}")
print(f"H at its sum/product-row tokens    -> 'left of GEQ': {p('row_toks'):5.1%}")
