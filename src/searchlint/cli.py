"""searchlint: lint for retrieval.

  searchlint init [--docs ./docs]     write searchlint.yaml
  searchlint index                    chunk + embed + index (incremental embeddings)
  searchlint golden draft N           LLM drafts N golden queries (status: draft) -> review golden.yaml
  searchlint golden approve           mark every draft as approved
  searchlint run [--ci] [--baseline]  evaluate; --ci exits 1 when the gate fails; --baseline stores this run as the baseline
  searchlint serve --mcp              MCP server over the same index
"""
import sys, os, argparse

def bar(i, n):
    w = 24; f = int(w * i / max(1, n))
    sys.stderr.write(f"\rembed {i}/{n} ▕{'█' * f}{' ' * (w - f)}▏"); sys.stderr.flush()
    if i >= n: sys.stderr.write("\n")

def main(argv=None):
    ap = argparse.ArgumentParser(prog="searchlint", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="searchlint.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("--docs", default="./docs")
    sub.add_parser("index")
    g = sub.add_parser("golden"); gs = g.add_subparsers(dest="gcmd", required=True)
    gd = gs.add_parser("draft"); gd.add_argument("n", type=int, nargs="?", default=20)
    gs.add_parser("approve"); gs.add_parser("list")
    r = sub.add_parser("run"); r.add_argument("--ci", action="store_true"); r.add_argument("--baseline", action="store_true"); r.add_argument("--k", type=int)
    sv = sub.add_parser("serve"); sv.add_argument("--mcp", action="store_true")
    a = ap.parse_args(argv)

    from .config import Config
    if a.cmd == "init":
        if os.path.exists(a.config): print(f"{a.config} exists"); return 1
        Config.write_default(a.config, a.docs); print(f"wrote {a.config}  (docs: {a.docs}). next: searchlint index"); return 0
    cfg = Config(a.config)

    if a.cmd == "index":
        from .ingest import build_chunks
        from .index import Index
        files, chunks = build_chunks(cfg)
        print(f"{len(files)} files -> {len(chunks)} chunks")
        Index(cfg, chunks=chunks, progress=bar)
        import json; meta = json.load(open(os.path.join(cfg.state, "index.json")))
        print(f"indexed. embedded {meta['embedded_new']} new chunks with {meta['model']}"); return 0

    if a.cmd == "golden":
        from . import golden as G
        rows = G.load(cfg.golden_path)
        if a.gcmd == "list":
            for x in rows: print(f"[{x.get('status','ok')}] {x['q']}  -> {x['expect']}" + (f"  ({x['doc']})" if x.get('doc') else ""))
            print(f"{len(rows)} queries"); return 0
        if a.gcmd == "approve":
            n = sum(1 for x in rows if x.get("status") == "draft")
            for x in rows: x["status"] = "ok"
            G.save(cfg.golden_path, rows); print(f"approved {n}"); return 0
        if a.gcmd == "draft":
            from .index import Index
            from .llm import Ollama
            ix = Index(cfg); new = G.draft(cfg, Ollama(cfg["llm"]["host"]), ix.chunks, a.n, rows)
            G.save(cfg.golden_path, rows + new)
            print(f"drafted {len(new)} -> {cfg['golden']}. Review them, then: searchlint golden approve"); return 0

    if a.cmd == "run":
        from . import golden as G
        from .index import Index
        from .run import run
        rows = [x for x in G.load(cfg.golden_path) if x.get("status", "ok") != "draft"]
        if not rows: print("no approved golden queries. searchlint golden draft 20 -> review -> searchlint golden approve"); return 1
        if a.k: cfg.d["search"]["k"] = a.k
        ok, _ = run(cfg, Index(cfg), rows, ci=a.ci, update_baseline=a.baseline)
        return 0 if ok else 1

    if a.cmd == "serve":
        from .mcp_server import serve
        serve(a.config); return 0

if __name__ == "__main__":
    sys.exit(main())
