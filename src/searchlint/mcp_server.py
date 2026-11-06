"""MCP server (stdio): search over the linted index, report its last measured quality, log queries for golden candidates."""
import os, json, time, asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types
from .config import Config
from .index import Index

def serve(cfg_path="searchlint.yaml"):
    cfg = Config(cfg_path); index = Index(cfg)
    log_path = os.path.join(cfg.state, "queries.jsonl")
    server = Server("searchlint")

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(name="search", description="Search the project's documents (hybrid BM25 + dense). Returns the top chunks with document and page.",
                       inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": cfg["search"]["k"]}}, "required": ["query"]}),
            types.Tool(name="quality", description="How trustworthy this search is: the last searchlint run (recall@k, MRR, nDCG, n golden queries, when).",
                       inputSchema={"type": "object", "properties": {}}),
            types.Tool(name="recent_queries", description="Recent queries made through this server (candidates for new golden queries).",
                       inputSchema={"type": "object", "properties": {"limit": {"type": "integer", "default": 50}}}),
        ]

    @server.call_tool()
    async def call_tool(name, arguments):
        if name == "search":
            q, k = arguments["query"], int(arguments.get("k") or cfg["search"]["k"])
            hits = index.search(q, k=k)
            with open(log_path, "a") as f: f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "q": q, "top": [h["id"] for h in hits]}, ensure_ascii=False) + "\n")
            text = "\n\n".join(f"[{i+1}] {h['doc']}" + (f" p.{h['page']}" if h.get("page") else "") + f"\n{h['text']}" for i, h in enumerate(hits))
            return [types.TextContent(type="text", text=text or "(no results)")]
        if name == "quality":
            p = os.path.join(cfg.state, "last.json")
            if not os.path.exists(p): return [types.TextContent(type="text", text="no searchlint run yet: quality unknown")]
            r = json.load(open(p)); s = r["summary"]
            return [types.TextContent(type="text", text=f"searchlint run at {r['at']}: recall@{r['k']}={s['recall']:.3f} MRR={s['mrr']:.3f} nDCG={s['ndcg']:.3f} over {s['n']} golden queries ({r['mode']}).")]
        if name == "recent_queries":
            lim = int(arguments.get("limit") or 50)
            rows = [json.loads(l) for l in open(log_path)] if os.path.exists(log_path) else []
            return [types.TextContent(type="text", text="\n".join(f"{r['at']}  {r['q']}" for r in rows[-lim:]) or "(none)")]
        raise ValueError(name)

    async def main():
        async with stdio_server() as (r, w):
            await server.run(r, w, server.create_initialization_options())
    asyncio.run(main())
