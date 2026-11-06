"""MCP server (stdio): search over the linted index, report its last measured quality, log queries as golden candidates."""
import os, json, time
from mcp.server.mcpserver import MCPServer
from .config import Config
from .index import Index

def serve(cfg_path="searchlint.yaml"):
    cfg = Config(cfg_path); index = Index(cfg)
    log_path = os.path.join(cfg.state, "queries.jsonl")
    mcp = MCPServer("searchlint", instructions="Search the project's documents. Call quality first to learn how far to trust search results.")

    @mcp.tool(description="Search the project's documents (hybrid BM25 + dense). Returns the top chunks with document and page.")
    def search(query: str, k: int = cfg["search"]["k"]) -> str:
        hits = index.search(query, k=int(k))
        with open(log_path, "a") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "q": query, "top": [h["id"] for h in hits]}, ensure_ascii=False) + "\n")
        return "\n\n".join(f"[{i+1}] {h['doc']}" + (f" p.{h['page']}" if h.get("page") else "") + f"\n{h['text']}" for i, h in enumerate(hits)) or "(no results)"

    @mcp.tool(description="How trustworthy this search is: the last searchlint run (recall@k, MRR, nDCG, number of golden queries, when).")
    def quality() -> str:
        p = os.path.join(cfg.state, "last.json")
        if not os.path.exists(p): return "no searchlint run yet: quality unknown"
        r = json.load(open(p)); s = r["summary"]
        return f"searchlint run at {r['at']}: recall@{r['k']}={s['recall']:.3f} MRR={s['mrr']:.3f} nDCG={s['ndcg']:.3f} over {s['n']} golden queries ({r['mode']})."

    @mcp.tool(description="Recent queries made through this server: candidates for new golden queries.")
    def recent_queries(limit: int = 50) -> str:
        rows = [json.loads(l) for l in open(log_path)] if os.path.exists(log_path) else []
        return "\n".join(f"{r['at']}  {r['q']}" for r in rows[-int(limit):]) or "(none)"

    mcp.run("stdio")
