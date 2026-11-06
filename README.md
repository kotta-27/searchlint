# searchlint

日本語: [README.ja.md](README.ja.md)

Lint for retrieval. Golden queries live in your repo; `searchlint run --ci` fails the build when
recall drops after a document edit, a chunking change or an embedding-model swap. The same index is
served to agents over MCP, together with its measured quality.

Single Rust binary. Japanese segmentation (lindera, IPADIC embedded), BM25 (tantivy) and the BGE-M3 embeddings
(ONNX, `embed.provider: fastembed`) all run in-process. Nothing leaves the machine. Ollama is only needed for
`golden draft` (an LLM writes the first queries) or if you prefer `embed.provider: ollama`.
`search.mode: bm25` needs no embeddings at all.

```
cargo install --git https://github.com/kotta-27/searchlint   # or: cargo build --release
ollama pull qwen3:8b                                           # only for `golden draft`

searchlint init --docs ./docs        # writes searchlint.yaml
searchlint index                     # txt / md / pdf -> cleaned chunks -> FTS5 (+ FAISS unless bm25)
searchlint golden draft 20           # LLM drafts queries into golden.yaml (status: draft)
#   review golden.yaml: fix or delete drafts, then
searchlint golden approve
searchlint run                       # recall@k, MRR; the first passing run becomes searchlint.baseline.json
searchlint run --ci                  # exit 1 below the gate or if recall dropped > max_drop vs the baseline
searchlint serve --mcp               # MCP: search / quality / list_documents / recent_queries / reload
```

Exit codes: `0` ok, `1` gate failed, `2` cannot evaluate (stale index, bad config or golden, Ollama down).

## Files

| file | commit? | what |
|---|---|---|
| `searchlint.yaml` | yes | chunking, embedding, search mode, gate |
| `golden.yaml` | yes | the queries that must be found |
| `searchlint.baseline.json` | yes | the reference run; `max_drop` compares against it. Update with `searchlint run --baseline` and review the diff in the PR |
| `.searchlint/` | no | index, embedding cache (per model + chunk text), last run, MCP query log |

## golden.yaml

```yaml
- q: 添付ファイルの最大サイズは何MBですか？
  expect: ["添付ファイルの最大サイズ"]   # a hit = a retrieved chunk that contains any of these snippets
  doc: manual.pdf                       # optional: restrict to one document
  status: ok                            # draft rows are ignored by run
```

Snippets, not chunk ids: the golden set survives re-chunking, re-embedding and edits. Keep snippets
specific (4+ characters, ideally a phrase): a short common word matches everywhere and hides regressions.
`run` warns about short ones. Drafts are kept only when the answer really occurs in the chunk.

When a document is revised and the snippet itself changes (9MB -> 10MB), the query fails. `run` tells the two
cases apart: `[STALE GOLDEN]` means no chunk in the corpus contains the snippet any more (edit golden.yaml);
`[retrieval: rank 12]` means the snippet exists but was not in the top k (a retrieval regression).

## searchlint.yaml

```yaml
docs: ./docs
include: ["**/*.md", "**/*.txt", "**/*.pdf"]
chunk: {size: 500, overlap: 60, clean: true}   # clean: drop running headers, page numbers, TOC lines
embed: {provider: fastembed, model: bge-m3, device: cpu}   # in-process ONNX, model cached in ~/.cache/searchlint
#      device: gpu tries CoreML / CUDA / DirectML; on Apple silicon CPU is 10x faster for BGE-M3, so cpu is the default
#      provider: ollama (+ host) uses a running Ollama instead
llm:   {provider: ollama, model: qwen3:8b}      # only for golden drafting
search: {mode: hybrid, k: 5, pool: 30}          # bm25 | dense | hybrid (RRF)
gate:  {recall: 0.9, mrr: 0.0, max_drop: 0.02}
```

`run` refuses a stale index (config or documents changed since `index`): rerun `searchlint index`, or
`searchlint run --index` to rebuild on the fly. A baseline made with a different config is not compared
against (the drop check is skipped with a note); reset it with `--baseline`.

## How many golden queries

With 8 queries one miss is 0.125 recall: every gate fires. The gate gets meaningful around 40 and
comfortable at 100. Grow it from `recent_queries` (what agents actually ask) and from `golden draft`.

## CI

```yaml
# .github/workflows/searchlint.yml
- run: pip install searchlint && searchlint index && searchlint run --ci
```

Hybrid or dense mode with `provider: fastembed` runs on any runner (CPU, no server); the model (~600 MB) and
`.searchlint/embed-cache/` are worth caching between runs so re-indexing only embeds new or changed chunks.

`searchlint run --json` prints the summary, problems, notes and missed queries as JSON.

## MCP

```json
{ "mcpServers": { "searchlint": { "command": "searchlint", "args": ["serve", "--mcp"], "cwd": "/path/to/repo" } } }
```

Every tool returns structured content.

- `search(query, k=5, doc=None)`: ranked chunks with `id`, `doc`, `page`, `text`. The description tells the agent the mode (bm25: use the documents' words, not paraphrases).
- `quality()`: `ok` (fresh index and recall above the gate), `recall`, `mrr`, `n`, `at`, `stale`, `reason`. Call it first.
- `list_documents()`: what is indexed, with chunk and page counts.
- `recent_queries(limit)`: distinct recent queries with hit counts: golden candidates.
- `reload()`: pick up a new index without restarting.

## Metrics

- recall@k: a golden query hits if any of the top k chunks contains an expected snippet
- MRR: 1 / rank of the first hit, averaged
- latency p50 / p95 / max: time per search over the golden queries (after one warm-up), the same path MCP uses.
  Mostly the query embedding (BGE-M3 on CPU, longer questions cost more); bm25 alone is ~1-3 ms.
  `gate.p95_ms` fails the run above a threshold (off by default: latency depends on the machine)
- startup: index open + embedding model load (paid once per MCP server). index build time is recorded in index.json
