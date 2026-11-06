"""Hybrid index: SQLite FTS5 (BM25 over janome tokens) + FAISS (inner product on normalized vectors), fused by RRF.
Embeddings are cached by text hash under the state dir, so re-indexing only embeds new or changed chunks."""
import os, json, sqlite3, threading, numpy as np, faiss
from janome.tokenizer import Tokenizer
from .llm import Ollama

_tok = Tokenizer()
def tokenize(text): return [t.surface for t in _tok.tokenize(text) if t.surface.strip()]

class Index:
    def __init__(self, cfg, chunks=None, progress=None):
        self.cfg = cfg; self.state = cfg.state; os.makedirs(self.state, exist_ok=True)
        self.chunks_path = os.path.join(self.state, "chunks.json")
        self.fts_path = os.path.join(self.state, "fts.sqlite")
        self.faiss_path = os.path.join(self.state, "dense.faiss")
        self.cache_path = os.path.join(self.state, "embed-cache.npz")
        self.ollama = Ollama(cfg["embed"]["host"])
        self._local = threading.local()
        if chunks is not None:
            self.chunks = chunks; self._build(progress)
        else:
            if not os.path.exists(self.chunks_path): raise FileNotFoundError("no index yet: run `searchlint index`")
            self.chunks = json.load(open(self.chunks_path))
        self.dense = faiss.read_index(self.faiss_path)

    # ---- build ----
    def _build(self, progress):
        json.dump(self.chunks, open(self.chunks_path, "w"), ensure_ascii=False)
        if os.path.exists(self.fts_path): os.remove(self.fts_path)
        db = sqlite3.connect(self.fts_path)
        db.execute("create virtual table doc using fts5(toks, tokenize='unicode61 remove_diacritics 0')")
        with db:
            db.executemany("insert into doc(rowid, toks) values (?, ?)", ((i, " ".join(tokenize(c["text"]))) for i, c in enumerate(self.chunks)))
        db.execute("insert into doc(doc) values ('optimize')"); db.commit(); db.close()
        # embeddings with a hash cache
        cache = {}
        if os.path.exists(self.cache_path):
            z = np.load(self.cache_path, allow_pickle=False); cache = dict(zip(z["keys"].tolist(), z["vecs"]))
        need = [c for c in self.chunks if c["sha"] not in cache]
        model = self.cfg["embed"]["model"]
        for i in range(0, len(need), 64):
            batch = need[i:i + 64]
            vecs = self.ollama.embed(model, [c["text"] for c in batch])
            for c, v in zip(batch, vecs): cache[c["sha"]] = v
            if progress: progress(min(i + 64, len(need)), len(need))
        keys = [c["sha"] for c in self.chunks]
        np.savez(self.cache_path, keys=np.array(list(cache.keys())), vecs=np.stack(list(cache.values())) if cache else np.zeros((0, 1)))
        mat = np.stack([cache[k] for k in keys]).astype(np.float32) if keys else np.zeros((0, 1024), np.float32)
        ix = faiss.IndexFlatIP(mat.shape[1]); ix.add(mat); faiss.write_index(ix, self.faiss_path)
        json.dump({"chunks": len(self.chunks), "embedded_new": len(need), "model": model}, open(os.path.join(self.state, "index.json"), "w"))

    # ---- query ----
    @property
    def db(self):
        if not hasattr(self._local, "db"): self._local.db = sqlite3.connect(f"file:{self.fts_path}?mode=ro", uri=True)
        return self._local.db

    def bm25(self, q, n):
        toks = [t.replace('"', "") for t in tokenize(q) if t.strip()]
        if not toks: return []
        rows = self.db.execute("select rowid from doc where doc match ? order by bm25(doc) limit ?", (" OR ".join(f'"{t}"' for t in toks), n)).fetchall()
        return [r[0] for r in rows]

    def dense_many(self, qs, n):
        _, ids = self.dense.search(self.ollama.embed(self.cfg["embed"]["model"], list(qs)).astype(np.float32), n)
        return [[int(i) for i in row if i >= 0] for row in ids]

    def search(self, q, k=None, mode=None, pool=None):
        s = self.cfg["search"]; k, mode, pool = k or s["k"], mode or s["mode"], pool or s["pool"]
        qs = [q] if isinstance(q, str) else list(dict.fromkeys(q))
        if len(qs) == 1 and mode == "bm25": idx = self.bm25(qs[0], k)
        elif len(qs) == 1 and mode == "dense": idx = self.dense_many(qs, k)[0]
        else:
            lists = []
            if mode in ("bm25", "hybrid"): lists += [self.bm25(x, pool) for x in qs]
            if mode in ("dense", "hybrid"): lists += self.dense_many(qs, pool)
            score = {}
            for lst in lists:
                for r, i in enumerate(lst): score[i] = score.get(i, 0) + 1 / (60 + r)
            idx = sorted(score, key=score.get, reverse=True)[:k]
        return [self.chunks[i] for i in idx]
