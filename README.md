# searchlint

Lint for retrieval. Golden queries live in your repo; `searchlint run --ci` fails the build when
recall drops after a document edit, a chunking change or an embedding-model swap. The same index is
served to agents over MCP, together with its last measured quality.

Local only: embeddings and golden drafting go through Ollama. Nothing leaves the machine.

```
pip install searchlint        # (or: pip install -e . from this repo)
ollama pull bge-m3 qwen3:8b

searchlint init --docs ./docs        # writes searchlint.yaml
searchlint index                     # txt / md / pdf -> cleaned chunks -> FTS5 + FAISS
searchlint golden draft 20           # LLM drafts queries into golden.yaml (status: draft)
#   review golden.yaml, fix or delete drafts, then
searchlint golden approve
searchlint run                       # recall@k, MRR, nDCG; first run becomes the baseline
searchlint run --ci                  # exit 1 if below the gate or if recall dropped > max_drop
searchlint serve --mcp               # MCP: search / quality / recent_queries
```

## golden.yaml

```yaml
- q: 添付ファイルの最大サイズは何MBですか？
  expect: ["9MB"]            # a hit = a retrieved chunk that contains any of these snippets
  doc: manual.pdf             # optional: restrict to one document
```

Snippets, not chunk ids: the golden set survives re-chunking, re-embedding and edits.

## searchlint.yaml

```yaml
docs: ./docs
include: ["**/*.md", "**/*.txt", "**/*.pdf"]
chunk: {size: 500, overlap: 60, clean: true}   # clean: drop running headers, page numbers, TOC lines
embed: {provider: ollama, model: bge-m3}
llm:   {provider: ollama, model: qwen3:8b}      # only for golden drafting
search: {mode: hybrid, k: 5, pool: 30}          # bm25 | dense | hybrid (RRF)
gate:  {recall: 0.9, mrr: 0.0, max_drop: 0.02}
```

## CI

```yaml
# .github/workflows/searchlint.yml (self-hosted runner with Ollama)
- run: pip install searchlint && searchlint index && searchlint run --ci
```

State (`.searchlint/`) holds the index, an embedding cache keyed by chunk text, the baseline and the
last run. Commit `golden.yaml` and `searchlint.yaml`; ignore `.searchlint/`.

## MCP

```json
{ "mcpServers": { "searchlint": { "command": "searchlint", "args": ["serve", "--mcp"], "cwd": "/path/to/repo" } } }
```

`quality` returns the last run (recall@k, MRR, n, when) so an agent knows how far to trust `search`.
`recent_queries` lists what agents asked: candidates for new golden queries.

## Metrics

- recall@k: a golden query hits if any of the top k chunks contains an expected snippet
- MRR: 1 / rank of the first hit
- nDCG: rank-weighted hits, binary relevance
