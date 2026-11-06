"""searchlint: lint for retrieval.

  searchlint init [--docs ./docs]        write searchlint.yaml
  searchlint index                       chunk + embed + index (incremental embeddings; bm25 mode needs no embeddings)
  searchlint golden draft N [--seed S]   LLM drafts N golden queries (status: draft) -> review golden.yaml
  searchlint golden approve              mark every draft as ok
  searchlint golden list
  searchlint run [--ci] [--baseline] [--index] [--json]
                                         evaluate. --ci: exit 1 when the gate fails. --baseline: store this run as the
                                         baseline. --index: rebuild the index first if it is stale. --json: machine output
  searchlint serve --mcp                 MCP server over the same index

exit codes: 0 ok, 1 gate failed, 2 cannot evaluate (stale index, bad config, Ollama down)
"""
import sys, os, json, argparse

def bar(i, n):
    w = 24; f = int(w * i / max(1, n))
    sys.stderr.write(f"\rembed {i}/{n} ▕{'█' * f}{' ' * (w - f)}▏"); sys.stderr.flush()
    if i >= n: sys.stderr.write("\n")

def die(msg, code=2):
    print(f"searchlint: {msg}", file=sys.stderr); return code

def main(argv=None):
    ap = argparse.ArgumentParser(prog="searchlint", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="searchlint.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("--docs", default="./docs")
    sub.add_parser("index")
    g = sub.add_parser("golden"); gs = g.add_subparsers(dest="gcmd", required=True)
    gd = gs.add_parser("draft"); gd.add_argument("n", type=int, nargs="?", default=20); gd.add_argument("--seed", type=int)
    gs.add_parser("approve"); gs.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--ci", action="store_true"); r.add_argument("--baseline", action="store_true")
    r.add_argument("--index", action="store_true", help="rebuild the index first if it is stale"); r.add_argument("--json", action="store_true")
    sv = sub.add_parser("serve"); sv.add_argument("--mcp", action="store_true")
    a = ap.parse_args(argv)

    from .config import Config, ConfigError
    if a.cmd == "init":
        if os.path.exists(a.config): return die(f"{a.config} exists")
        Config.write_default(a.config, a.docs); print(f"wrote {a.config}  (docs: {a.docs}). next: searchlint index"); return 0
    try: cfg = Config(a.config)
    except ConfigError as e: return die(str(e))

    try:
        if a.cmd == "index": return do_index(cfg)
        if a.cmd == "golden": return do_golden(cfg, a)
        if a.cmd == "run": return do_run(cfg, a)
        if a.cmd == "serve":
            from .mcp_server import serve
            serve(a.config); return 0
    except Exception as e:  # Ollama down, broken yaml, missing index: never a traceback in CI
        import httpx
        from .golden import GoldenError
        from .index import IndexError_
        if isinstance(e, (httpx.HTTPError, ConnectionError)): return die(f"cannot reach Ollama at {cfg['embed']['host']}: {e}")
        if isinstance(e, (GoldenError, IndexError_)): return die(str(e))
        raise

def do_index(cfg):
    from .ingest import build_chunks
    from .index import Index
    files, chunks = build_chunks(cfg)
    if not chunks: return die(f"no chunks: no files matched {cfg['include']} under {cfg.docs}")
    print(f"{len(files)} files -> {len(chunks)} chunks")
    ix = Index(cfg, chunks=chunks, progress=bar)
    m = ix.meta
    print(f"indexed. " + (f"embedded {m['embedded_new']} new chunks with {m['model']}" if m["model"] else "bm25 only, no embeddings")); return 0

def do_golden(cfg, a):
    from . import golden as G
    rows = G.load(cfg.golden_path)
    if a.gcmd == "list":
        for x in rows: print(f"[{x['status']}] {x['q']}  -> {x['expect']}" + (f"  ({x['doc']})" if x.get("doc") else ""))
        for w in G.warnings(rows): print("warning:", w)
        print(f"{len(rows)} queries ({sum(1 for x in rows if x['status'] == 'draft')} draft)"); return 0
    if a.gcmd == "approve":
        n = sum(1 for x in rows if x["status"] == "draft")
        for x in rows: x["status"] = "ok"
        G.save(cfg.golden_path, rows); print(f"approved {n}"); return 0
    if a.gcmd == "draft":
        from .index import Index
        from .llm import Ollama
        ix = Index(cfg); new = G.draft(cfg, Ollama(cfg["llm"]["host"]), ix.chunks, a.n, rows, seed=a.seed)
        G.save(cfg.golden_path, rows + new)
        print(f"drafted {len(new)} -> {cfg['golden']}. Review them, then: searchlint golden approve"); return 0

def do_run(cfg, a):
    from . import golden as G
    from .index import Index
    from .run import run
    rows = [x for x in G.load(cfg.golden_path) if x["status"] != "draft"]
    if not rows: return die("no approved golden queries. searchlint golden draft 20 -> review -> searchlint golden approve")
    ix = Index(cfg)
    stale = ix.staleness()
    if stale and a.index:
        print(f"index stale ({stale}): rebuilding"); do_index(cfg); ix = Index(cfg)
    elif stale:
        return die(f"index is stale: {stale}. Run `searchlint index` (or `searchlint run --index`)")
    lines = []
    ok, result = run(cfg, ix, rows, ci=a.ci, update_baseline=a.baseline, out=(lines.append if a.json else print))
    if a.json:
        print(json.dumps({k: v for k, v in result.items() if k != "per"} | {"missed": [p["q"] for p in result["per"] if not p["hit"]]}, ensure_ascii=False, indent=1))
    return 0 if ok or not a.ci else 1

if __name__ == "__main__":
    sys.exit(main())
