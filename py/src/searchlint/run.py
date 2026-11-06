"""Evaluate the index against golden.yaml: recall@k and MRR per query; compare with the committed baseline; gate.

Exit codes (cli): 0 pass, 1 gate failed, 2 cannot evaluate (stale index, config/env error)."""
import os, json, time
from . import golden as G

def evaluate(index, rows, k):
    per = []
    for r in rows:
        chunks = index.search(r["q"], k=k)
        hits = [G.hit(c, r) for c in chunks]
        rank = next((i for i, h in enumerate(hits) if h), None)
        per.append({"q": r["q"], "hit": rank is not None, "rank": None if rank is None else rank + 1,
                    "mrr": 0.0 if rank is None else 1 / (rank + 1),
                    "top": [{"id": c["id"], "doc": c["doc"], "page": c.get("page"), "text": c["text"][:120]} for c in chunks[:3]]})
    n = len(per) or 1
    return {"n": len(per), "recall": sum(p["hit"] for p in per) / n, "mrr": sum(p["mrr"] for p in per) / n}, per

def load_json(p): return json.load(open(p)) if os.path.exists(p) else None

def run(cfg, index, rows, ci=False, update_baseline=False, out=print):
    k = int(cfg["search"]["k"])
    summary, per = evaluate(index, rows, k)
    result = {"at": time.strftime("%Y-%m-%dT%H:%M"), "k": k, "mode": cfg["search"]["mode"], "fingerprint": cfg.fingerprint(),
              "index_built": index.meta.get("built"), "summary": summary, "per": per}
    os.makedirs(cfg.state, exist_ok=True)
    json.dump(result, open(os.path.join(cfg.state, "last.json"), "w"), ensure_ascii=False, indent=1)

    base = load_json(cfg.baseline_path)
    comparable = bool(base) and base.get("fingerprint") == result["fingerprint"]
    gate, problems, notes = cfg["gate"], [], []

    out(f"searchlint  n={summary['n']}  k={k}  mode={cfg['search']['mode']}")
    def line(name, v):
        s = f"  {name:10} {v:.3f}"
        if comparable: s += f"   (baseline {base['summary'][name]:.3f}, {v - base['summary'][name]:+.3f})"
        out(s)
    line("recall", summary["recall"]); line("mrr", summary["mrr"])
    for w in G.warnings(rows)[:5]: notes.append("warning: " + w)
    misses = [p for p in per if not p["hit"]]
    if misses:
        out(f"  missed {len(misses)}:")
        for p in misses[:10]:
            t = p["top"][0] if p["top"] else None
            out(f"    - {p['q'][:60]}   top1: " + (f"{t['doc']}" + (f" p.{t['page']}" if t.get("page") else "") if t else "-"))
    if base and not comparable:
        notes.append("baseline was made with a different config (chunk / embed / search): drop check skipped. Run `searchlint run --baseline` to reset it.")
    if comparable:
        was = {p["q"]: p for p in base["per"]}
        broken = [p for p in per if not p["hit"] and was.get(p["q"], {}).get("hit")]
        fixed = [p for p in per if p["hit"] and was.get(p["q"]) and not was[p["q"]]["hit"]]
        if broken or fixed: out(f"  vs baseline: {len(fixed)} fixed, {len(broken)} broken" + (": " + "; ".join(p["q"][:40] for p in broken[:5]) if broken else ""))
        if summary["recall"] < base["summary"]["recall"] - float(gate.get("max_drop", 0.02)):
            problems.append(f"recall dropped {base['summary']['recall'] - summary['recall']:.3f} (> max_drop {gate.get('max_drop', 0.02)})")
    if summary["recall"] < float(gate["recall"]): problems.append(f"recall {summary['recall']:.3f} < gate.recall {gate['recall']}")
    if summary["mrr"] < float(gate.get("mrr", 0)): problems.append(f"mrr {summary['mrr']:.3f} < gate.mrr {gate['mrr']}")
    if summary["n"] < 20: notes.append(f"only {summary['n']} golden queries: one miss = {1 / max(1, summary['n']):.2f} recall. The gate gets meaningful around 40+.")
    for n_ in notes: out("  " + n_)

    ok = not problems
    if ci:
        for p in problems: out("  FAIL " + p)
        out("  PASS" if ok else "  gate failed")
    if update_baseline or (base is None and ok):
        json.dump(result, open(cfg.baseline_path, "w"), ensure_ascii=False, indent=1)
        out(f"  baseline saved -> {os.path.relpath(cfg.baseline_path, cfg.root)}" + ("" if update_baseline else " (first passing run; commit it)"))
    result["problems"], result["notes"], result["ok"] = problems, notes, ok
    return ok, result
