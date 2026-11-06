"""MCP server (stdio): search over the linted index, report its measured quality, list documents, log queries as golden candidates.
Tool results are structured (dict) so agents can read fields instead of parsing text."""
import os, json, time
from typing import Annotated
from pydantic import Field
from mcp.server.mcpserver import MCPServer
from .config import Config
from .index import Index

def serve(cfg_path="searchlint.yaml"):
    cfg = Config(cfg_path)
    state = {"index": Index(cfg)}
    log_path = os.path.join(cfg.state, "queries.jsonl")
    mode = cfg["search"]["mode"]
    hint = {"bm25": "Keyword search: use the exact words the documents would use, not paraphrases.",
            "dense": "Semantic search: natural-language questions work; exact identifiers may be missed.",
            "hybrid": "Keyword + semantic search fused by rank."}[mode]
    mcp = MCPServer("searchlint", instructions="Search the project's documents. Call quality first: it says whether the index is fresh and how often golden queries are found.")

    def quality_dict():
        p = os.path.join(cfg.state, "last.json"); ix = state["index"]
        stale = ix.staleness()
        if not os.path.exists(p):
            return {"ok": False, "measured": False, "stale": bool(stale), "reason": "no searchlint run yet" + (f"; {stale}" if stale else "")}
        r = json.load(open(p)); s = r["summary"]
        reasons = []
        if stale: reasons.append(stale)
        if r.get("fingerprint") != cfg.fingerprint(): reasons.append("last run used a different config")
        if s["recall"] < float(cfg["gate"]["recall"]): reasons.append(f"recall {s['recall']:.3f} below gate {cfg['gate']['recall']}")
        return {"ok": not reasons, "measured": True, "stale": bool(stale), "at": r["at"], "k": r["k"], "mode": r["mode"], "n": s["n"],
                "recall": round(s["recall"], 3), "mrr": round(s["mrr"], 3), "gate_recall": cfg["gate"]["recall"], "reason": "; ".join(reasons) or None}

    @mcp.tool(description=f"Search the project's documents ({mode}). {hint} Returns ranked chunks with document, page and chunk id.")
    def search(query: Annotated[str, Field(min_length=1, max_length=500)],
               k: Annotated[int, Field(ge=1, le=20)] = int(cfg["search"]["k"]),
               doc: Annotated[str | None, Field(description="restrict to one document path as returned by list_documents")] = None) -> dict:
        q = query.strip()
        if not q: return {"query": query, "mode": mode, "hits": [], "note": "empty query"}
        hits = state["index"].search(q, k=k, doc=doc)
        with open(log_path, "a") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "q": q, "n_hits": len(hits), "top": [h["id"] for h in hits[:3]]}, ensure_ascii=False) + "\n")
        return {"query": q, "mode": mode, "k": k, "doc": doc,
                "hits": [{"rank": i + 1, "id": h["id"], "doc": h["doc"], "page": h.get("page"), "text": h["text"]} for i, h in enumerate(hits)]}

    @mcp.tool(description="Whether to trust search: ok (fresh index + measured recall above the gate), recall@k, MRR, n golden queries, when measured, and the reason if not ok.")
    def quality() -> dict:
        return quality_dict()

    @mcp.tool(description="Documents in the index with chunk and page counts. Use it to tell 'no such document' from 'wrong words'.")
    def list_documents() -> dict:
        m = state["index"].meta
        return {"built": m.get("built"), "documents": state["index"].documents()}

    @mcp.tool(description="Recent queries made through this server with hit counts: candidates for new golden queries.")
    def recent_queries(limit: Annotated[int, Field(ge=1, le=500)] = 50) -> dict:
        rows = [json.loads(l) for l in open(log_path)] if os.path.exists(log_path) else []
        seen = {}
        for r in rows: seen[r["q"]] = {"q": r["q"], "last": r["at"], "n_hits": r.get("n_hits"), "count": seen.get(r["q"], {}).get("count", 0) + 1}
        return {"queries": list(seen.values())[-limit:]}

    @mcp.tool(description="Reload the index from disk after `searchlint index` ran.")
    def reload() -> dict:
        state["index"] = Index(cfg); m = state["index"].meta
        return {"built": m.get("built"), "chunks": m.get("chunks")}

    mcp.run("stdio")
