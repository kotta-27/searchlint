# searchlint

検索（retrieval）の lint。正解クエリをリポジトリに置き、`searchlint run --ci` が「文書の更新・チャンク設定の変更・
埋め込みモデルの差し替えで recall が落ちた」ときにビルドを止める。同じ索引を MCP でエージェントに配り、
測った品質も一緒に返す。

English: [README.md](README.md)

Rust の単一バイナリ。日本語の分かち書き（lindera、IPADIC 同梱）、BM25（tantivy）、BGE-M3 の埋め込み
（ONNX、`embed.provider: fastembed`）がすべてプロセス内で動く。外に何も送らない。
Ollama が要るのは `golden draft`（LLM に最初のクエリを下書きさせる）か、`embed.provider: ollama` を選んだときだけ。
`search.mode: bm25` なら埋め込み自体が不要。

```
cargo install --git https://github.com/kotta-27/searchlint   # または cargo build --release
ollama pull qwen3:8b                                           # golden draft にだけ必要

searchlint init --docs ./docs        # searchlint.yaml を書く
searchlint index                     # txt / md / pdf → 掃除 → チャンク → tantivy（bm25 以外は + 埋め込み）
searchlint golden draft 20           # LLM が golden.yaml に下書き（status: draft）
#   golden.yaml を見直す: 直す・消す。そのあと
searchlint golden approve
searchlint run                       # recall@k と MRR。最初に合格した run が searchlint.baseline.json になる
searchlint run --ci                  # 合格ライン未満、または基準から max_drop 超で落ちたら exit 1
searchlint serve --mcp               # MCP: search / quality / list_documents / recent_queries / reload
```

終了コード: `0` 合格、`1` ゲート失敗、`2` 評価できない（索引が古い、設定や golden が壊れている、Ollama 停止）。

## ファイル

| ファイル | コミット | 中身 |
|---|---|---|
| `searchlint.yaml` | する | チャンク、埋め込み、検索方式、合格ライン |
| `golden.yaml` | する | 必ず見つかるべきクエリ |
| `searchlint.baseline.json` | する | 基準となる run。`max_drop` はこれと比べる。更新は `searchlint run --baseline`、差分は PR で読む |
| `.searchlint/` | しない | 索引、埋め込みキャッシュ（モデル名 + チャンク本文ごと）、最後の結果、MCP の質問ログ |

## golden.yaml

```yaml
- q: 添付ファイルの最大サイズは何MBですか？
  expect: ["添付ファイルの最大サイズ"]   # 取れたチャンクのどれかにこの文字列が含まれていればヒット
  doc: manual.pdf                       # 任意: この文書に限定
  status: ok                            # draft の行は run が無視する
```

期待するのは chunk id ではなく「含まれるべき文字列」。再チャンク・再埋め込み・文書の修正で壊れない。
文字列は具体的に（4 文字以上、できれば語句）。短い一般語はどこにでも当たり、劣化を隠す。`run` が短いものに警告する。
下書きは「答えが本当にチャンクに含まれる」ものだけ残る。

文書が改訂されて文字列そのものが変わったとき（9MB → 10MB）はクエリが外れる。`run` はその 2 つを区別する:
`[STALE GOLDEN]` はコーパスのどのチャンクにも文字列が無い（golden.yaml を直す）、`[retrieval: rank 12]` は文字列は
あるが上位 k に入らなかった（検索の劣化）。

## searchlint.yaml

```yaml
docs: ./docs
include: ["**/*.md", "**/*.txt", "**/*.pdf"]
chunk: {size: 500, overlap: 60, clean: true}   # clean: ヘッダ・フッタ、ページ番号、目次行を除去
embed: {provider: fastembed, model: bge-m3, device: cpu}   # プロセス内 ONNX。モデルは ~/.cache/searchlint に 1 回だけ取得
#      device: gpu は CoreML / CUDA / DirectML を試す。Apple シリコンでは BGE-M3 は CPU の方が 10 倍速いので cpu が既定
#      provider: ollama（+ host）なら起動中の Ollama を使う
llm:   {provider: ollama, model: qwen3:8b}      # golden draft 専用
search: {mode: hybrid, k: 5, pool: 30}          # bm25 | dense | hybrid（RRF）
gate:  {recall: 0.9, mrr: 0.0, max_drop: 0.02}
```

`run` は古い索引（`index` 以後に設定か文書が変わった）を拒否する。`searchlint index` をやり直すか、
`searchlint run --index` でその場で再構築。基準が別の設定で作られていれば比較せず注記だけ出す（`--baseline` でリセット）。

## 正解クエリはいくつ要るか

8 問だと 1 問の外れが recall 0.125。ゲートはすべて発火する。意味を持つのは 40 問あたりから、100 問で安心。
`recent_queries`（エージェントが実際に聞いたこと）と `golden draft` で増やす。

## CI

```yaml
# .github/workflows/searchlint.yml
- run: cargo install --git https://github.com/kotta-27/searchlint && searchlint index && searchlint run --ci
```

`provider: fastembed` なら hybrid / dense でもどの runner でも動く（CPU、サーバ不要）。モデル（約 600 MB）と
`.searchlint/embed-cache/` を runner のキャッシュに残すと、再索引は新規・変更チャンクだけ埋め込む。

`searchlint run --json` は要約・問題・注記・外れたクエリを JSON で出す。

## MCP

```json
{ "mcpServers": { "searchlint": { "command": "searchlint", "args": ["serve", "--mcp"], "cwd": "/path/to/repo" } } }
```

全ツールが構造化された結果を返す。

- `search(query, k=5, doc=None)`: 順位付きチャンク（`id`, `doc`, `page`, `text`）。説明文に検索方式が入る（bm25 なら「言い換えでなく文書の語で」）
- `quality()`: `ok`（索引が新しく、recall が合格ライン以上）、`recall`、`mrr`、`n`、`at`、`stale`、`reason`。最初に呼ぶ
- `list_documents()`: 索引にある文書と、チャンク数・ページ数
- `recent_queries(limit)`: 最近の質問（重複除去、ヒット数つき）。golden の候補
- `reload()`: 再起動せずに新しい索引を読み込む

## 指標

- recall@k: 上位 k 件のどれかに期待する文字列が含まれればヒット
- MRR: 最初のヒットの順位の逆数の平均

## 構成

`src/` が Rust 実装（配布するもの）。`py/` は最初の Python プロトタイプで、参考用に残している。
