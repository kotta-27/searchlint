"""golden.yaml: the queries a retriever must get right.

- q: 納税証明書の交付請求の手順
  expect:                      # a hit counts when a retrieved chunk contains ANY of these snippets
    - 納税証明書の交付請求
  doc: manual.pdf               # optional: the chunk must come from this document
  note: drafted by LLM, approved 2026-11-07

Snippets, not chunk ids: the golden set survives re-chunking, re-embedding and document edits."""
import os, re, yaml, random, time

def load(path):
    if not os.path.exists(path): return []
    rows = yaml.safe_load(open(path)) or []
    for r in rows:
        r.setdefault("expect", []); r["expect"] = [str(x) for x in (r["expect"] if isinstance(r["expect"], list) else [r["expect"]])]
    return rows

def save(path, rows):
    with open(path, "w") as f:
        f.write("# searchlint golden queries. expect = snippets that must appear in a retrieved chunk.\n")
        yaml.safe_dump(rows, f, allow_unicode=True, sort_keys=False, width=120)

def norm(s): return re.sub(r"\s+", "", s or "")

def hit(chunk, row):
    if row.get("doc") and chunk["doc"] != row["doc"]: return False
    t = norm(chunk["text"])
    return any(norm(x) and norm(x) in t for x in row["expect"])

DRAFT = """次の資料だけから答えられる質問を1つ作ってください。資料固有の事実（固有名詞・数値・手順など）を問う短い質問にします。
answer は資料中にそのまま出てくる語句（2〜30文字）にしてください。
# 資料
{text}
JSON のみ: {{"question": "質問文", "answer": "資料中の語句"}}"""

def draft(cfg, ollama, chunks, n, existing):
    """LLM drafts n queries; each is kept only if its answer snippet really occurs in the chunk."""
    done = {r.get("src") for r in existing}
    cands = [c for c in chunks if c["id"] not in done and len(c["text"]) > 120]
    random.Random(int(time.time())).shuffle(cands)
    out, model = [], cfg["llm"]["model"]
    for c in cands:
        if len(out) >= n: break
        j = ollama.chat_json(model, DRAFT.format(text=c["text"][:1500]), max_tokens=200)
        if not j or not j.get("question") or not j.get("answer"): continue
        ans = str(j["answer"]).strip()
        if not (2 <= len(ans) <= 60) or norm(ans) not in norm(c["text"]): continue
        out.append({"q": j["question"].strip(), "expect": [ans], "doc": c["doc"], "src": c["id"], "status": "draft"})
    return out
