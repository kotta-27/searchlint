"""Evaluate the index against golden.yaml: recall@k, MRR, nDCG per query; compare with the stored baseline; gate."""
import os, json, math, time, hashlib
from . import golden as G

def ndcg(hits):
    """binary relevance; ideal ordering puts every hit first"""
    dcg = sum(1 / math.log2(i + 2) for i, h in enumerate(hits) if h)
    idcg = sum(1 / math.log2(i + 2) for i in range(sum(hits)))
    return dcg / idcg if idcg else 0.0

def evaluate(index, rows, k=None):
    per = []
    for r in rows:
        chunks = index.search(r["q"], k=k)
        hits = [G.hit(c, r) for c in chunks]
        rank = next((i for i, h in enumerate(hits) if h), None)
        per.append({"q": r["q"], "hit": rank is not None, "rank": None if rank is None else rank + 1,
                    "mrr": 0.0 if rank is None else 1 / (rank + 1), "ndcg": ndcg(hits),
                    "top": [{"id": c["id"], "doc": c["doc"], "page": c.get("page"), "text": c["text"][:120]} for c in chunks[:3]]})
    n = len(per) or 1
    summary = {"n": len(per), "recall": sum(p["hit"] for p in per) / n, "mrr": sum(p["mrr"] for p in per) / n, "ndcg": sum(p["ndcg"] for p in per) / n}
    return summary, per

def fingerprint(cfg, rows):
    h = hashlib.sha256(json.dumps({"chunk": cfg["chunk"], "embed": cfg["embed"]["model"], "search": cfg["search"], "n": len(rows)}, sort_keys=True).encode()).hexdigest()[:10]
    return h

def load_json(p): return json.load(open(p)) if os.path.exists(p) else None

def run(cfg, index, rows, ci=False, update_baseline=False, out=print):
    k = cfg["search"]["k"]
    summary, per = evaluate(index, rows, k)
    state = cfg.state; os.makedirs(state, exist_ok=True)
    last_path, base_path = os.path.join(state, "last.json"), os.path.join(state, "baseline.json")
    base = load_json(base_path)
    result = {"at": time.strftime("%Y-%m-%dT%H:%M"), "k": k, "mode": cfg["search"]["mode"], "config": fingerprint(cfg, rows), "summary": summary, "per": per}
    json.dump(result, open(last_path, "w"), ensure_ascii=False, indent=1)

    out(f"searchlint  n={summary['n']}  k={k}  mode={cfg['search']['mode']}")
    out(f"  recall@{k}  {summary['recall']:.3f}" + (f"   (baseline {base['summary']['recall']:.3f}, {summary['recall'] - base['summary']['recall']:+.3f})" if base else ""))
    out(f"  MRR        {summary['mrr']:.3f}" + (f"   (baseline {base['summary']['mrr']:.3f}, {summary['mrr'] - base['summary']['mrr']:+.3f})" if base else ""))
    out(f"  nDCG       {summary['ndcg']:.3f}")
    misses = [p for p in per if not p["hit"]]
    if misses:
        out(f"  missed {len(misses)}:")
        for p in misses[:10]:
            out(f"    - {p['q'][:60]}   top1: {p['top'][0]['doc'] if p['top'] else '-'}" + (f" p.{p['top'][0]['page']}" if p['top'] and p['top'][0].get('page') else ""))
    if base:
        was = {p["q"]: p for p in base["per"]}
        lost = [p for p in per if not p["hit"] and was.get(p["q"], {}).get("hit")]
        won = [p for p in per if p["hit"] and was.get(p["q"]) and not was[p["q"]]["hit"]]
        if lost or won: out(f"  vs baseline: {len(won)} fixed, {len(lost)} broken" + (": " + "; ".join(p["q"][:40] for p in lost[:5]) if lost else ""))

    ok = True
    gate = cfg["gate"]
    if ci:
        if summary["recall"] < gate["recall"]: out(f"  FAIL recall {summary['recall']:.3f} < gate {gate['recall']}"); ok = False
        if summary["mrr"] < gate.get("mrr", 0): out(f"  FAIL mrr {summary['mrr']:.3f} < gate {gate['mrr']}"); ok = False
        if base and summary["recall"] < base["summary"]["recall"] - gate.get("max_drop", 0.02):
            out(f"  FAIL recall dropped {base['summary']['recall'] - summary['recall']:.3f} > max_drop {gate.get('max_drop', 0.02)}"); ok = False
        out("  PASS" if ok else "  gate failed")
    if update_baseline or base is None:
        json.dump(result, open(base_path, "w"), ensure_ascii=False, indent=1); out("  baseline saved" if update_baseline else "  first run: saved as baseline")
    return ok, result
