"""golden.yaml: the queries a retriever must get right.

- q: 納税証明書の交付請求の手順
  expect:                      # a hit = a retrieved chunk that contains ANY of these snippets (whitespace-insensitive)
    - 納税証明書の交付請求
  doc: manual.pdf               # optional: the chunk must come from this document
  status: ok                    # draft rows are skipped by `run`

Snippets, not chunk ids: the golden set survives re-chunking, re-embedding and document edits.
Keep snippets specific (>= 4 characters, ideally a phrase): a short common word matches everywhere and hides regressions."""
import os, re, yaml, random, time

MIN_EXPECT = 4

class GoldenError(Exception): pass

def norm(s): return re.sub(r"\s+", "", str(s or ""))

def load(path):
    if not os.path.exists(path): return []
    rows = yaml.safe_load(open(path)) or []
    if not isinstance(rows, list): raise GoldenError(f"{path}: top level must be a list")
    out = []
    for i, r in enumerate(rows, 1):
        if not isinstance(r, dict) or not r.get("q"): raise GoldenError(f"{path}: entry {i} needs a `q` (got {r!r:.60})")
        exp = r.get("expect", [])
        exp = [str(x) for x in (exp if isinstance(exp, list) else [exp]) if str(x).strip()]
        if not exp: raise GoldenError(f"{path}: entry {i} ({r['q'][:30]}) needs `expect`")
        r = dict(r); r["expect"] = exp; r.setdefault("status", "ok"); out.append(r)
    return out

def warnings(rows):
    return [f"expect too short ({x!r}) in: {r['q'][:40]}" for r in rows for x in r["expect"] if len(norm(x)) < MIN_EXPECT]

def save(path, rows):
    with open(path, "w") as f:
        f.write("# searchlint golden queries. expect = snippets that must appear in a retrieved chunk. Review drafts, then `searchlint golden approve`.\n")
        yaml.safe_dump(rows, f, allow_unicode=True, sort_keys=False, width=120)

def hit(chunk, row):
    if row.get("doc") and chunk["doc"] != row["doc"]: return False
    t = norm(chunk["text"])
    return any(norm(x) in t for x in row["expect"] if norm(x))

DRAFT = """次の資料だけから答えられる質問を1つ作ってください。資料固有の事実（固有名詞・数値・手順など）を問う短い質問にします。
answer は資料中にそのまま出てくる語句（4〜30文字）にしてください。
# 資料
{text}
JSON のみ: {{"question": "質問文", "answer": "資料中の語句"}}"""

def draft(cfg, ollama, chunks, n, existing, seed=None):
    """LLM drafts n queries; each is kept only if its answer snippet really occurs in the chunk."""
    done = {r.get("src") for r in existing}
    cands = [c for c in chunks if c["id"] not in done and len(c["text"]) > 120]
    random.Random(seed if seed is not None else int(time.time())).shuffle(cands)
    out, model = [], cfg["llm"]["model"]
    for c in cands:
        if len(out) >= n: break
        j = ollama.chat_json(model, DRAFT.format(text=c["text"][:1500]), max_tokens=200)
        if not j or not j.get("question") or not j.get("answer"): continue
        ans = str(j["answer"]).strip()
        if not (MIN_EXPECT <= len(norm(ans)) <= 60) or norm(ans) not in norm(c["text"]): continue
        out.append({"q": j["question"].strip(), "expect": [ans], "doc": c["doc"], "src": c["id"], "status": "draft"})
    return out
