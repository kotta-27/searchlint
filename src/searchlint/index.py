"""Index: SQLite FTS5 (BM25 over janome tokens) and, unless search.mode is bm25, FAISS over normalized embeddings.
Embeddings are cached per (model, chunk text) under the state dir, so re-indexing only embeds new or changed chunks.
index.json records the config fingerprint and the documents manifest so `run` can tell when the index is stale."""
import os, json, time, sqlite3, numpy as np
from janome.tokenizer import Tokenizer
from .llm import Ollama
from .ingest import docs_manifest

_tok = Tokenizer()
def tokenize(text): return [t.surface for t in _tok.tokenize(text) if t.surface.strip()]

class IndexError_(Exception): pass

class Index:
    def __init__(self, cfg, chunks=None, progress=None):
        self.cfg = cfg; self.state = cfg.state; os.makedirs(self.state, exist_ok=True)
        self.chunks_path = os.path.join(self.state, "chunks.json")
        self.fts_path = os.path.join(self.state, "fts.sqlite")
        self.faiss_path = os.path.join(self.state, "dense.faiss")
        self.cache_path = os.path.join(self.state, "embed-cache.npz")
        self.meta_path = os.path.join(self.state, "index.json")
        self.ollama = Ollama(cfg["embed"]["host"]) if cfg.needs_embeddings else None
        self.dense = None
        if chunks is not None:
            self.chunks = chunks; self._build(progress)
        else:
            if not os.path.exists(self.meta_path): raise IndexError_("no index yet: run `searchlint index`")
            self.chunks = json.load(open(self.chunks_path))
        self.db = sqlite3.connect(f"file:{self.fts_path}?mode=ro", uri=True, check_same_thread=False)
        if cfg.needs_embeddings:
            import faiss
            if not os.path.exists(self.faiss_path): raise IndexError_("index was built without embeddings (bm25): run `searchlint index`")
            self.dense = faiss.read_index(self.faiss_path)

    @property
    def meta(self): return json.load(open(self.meta_path))

    def staleness(self):
        """None when fresh, else a reason"""
        m = self.meta
        if m.get("fingerprint") != self.cfg.fingerprint(): return "config changed since the index was built"
        if m.get("docs") != docs_manifest(self.cfg)[0]: return "documents changed since the index was built"
        return None

    # ---- build ----
    def _build(self, progress):
        json.dump(self.chunks, open(self.chunks_path, "w"), ensure_ascii=False)
        if os.path.exists(self.fts_path): os.remove(self.fts_path)
        db = sqlite3.connect(self.fts_path)
        db.execute("create virtual table doc using fts5(toks, tokenize='unicode61 remove_diacritics 0')")
        with db:
            db.executemany("insert into doc(rowid, toks) values (?, ?)", ((i, " ".join(tokenize(c["text"]))) for i, c in enumerate(self.chunks)))
        db.execute("insert into doc(doc) values ('optimize')"); db.commit(); db.close()
        embedded_new = 0
        if self.cfg.needs_embeddings:
            import faiss
            model = self.cfg["embed"]["model"]
            cache = {}
            if os.path.exists(self.cache_path):
                z = np.load(self.cache_path, allow_pickle=False); cache = dict(zip(z["keys"].tolist(), z["vecs"]))
            key = lambda c: f"{model}:{c['sha']}"
            need = [c for c in self.chunks if key(c) not in cache]
            for i in range(0, len(need), 64):
                batch = need[i:i + 64]
                for c, v in zip(batch, self.ollama.embed(model, [c["text"] for c in batch])): cache[key(c)] = v
                if progress: progress(min(i + 64, len(need)), len(need))
            embedded_new = len(need)
            # keep only this model's vectors in the cache file (one dimension per file)
            keep = {k: v for k, v in cache.items() if k.startswith(model + ":")}
            np.savez(self.cache_path, keys=np.array(list(keep.keys())), vecs=np.stack(list(keep.values())) if keep else np.zeros((0, 1)))
            mat = np.stack([cache[key(c)] for c in self.chunks]).astype(np.float32)
            ix = faiss.IndexFlatIP(mat.shape[1]); ix.add(mat); faiss.write_index(ix, self.faiss_path)
        elif os.path.exists(self.faiss_path):
            os.remove(self.faiss_path)
        docs_hash, files = docs_manifest(self.cfg)
        json.dump({"built": time.strftime("%Y-%m-%dT%H:%M"), "chunks": len(self.chunks), "files": len(files), "embedded_new": embedded_new,
                   "model": self.cfg["embed"]["model"] if self.cfg.needs_embeddings else None, "fingerprint": self.cfg.fingerprint(), "docs": docs_hash},
                  open(self.meta_path, "w"))

    # ---- query ----
    def bm25(self, q, n):
        toks = [t.replace('"', "") for t in tokenize(q) if t.strip()]
        if not toks: return []
        rows = self.db.execute("select rowid from doc where doc match ? order by bm25(doc), rowid limit ?", (" OR ".join(f'"{t}"' for t in toks), n)).fetchall()
        return [r[0] for r in rows]

    def dense_rank(self, q, n):
        _, ids = self.dense.search(self.ollama.embed(self.cfg["embed"]["model"], [q]).astype(np.float32), n)
        return [int(i) for i in ids[0] if i >= 0]

    def search(self, q, k=None, mode=None, pool=None, doc=None):
        s = self.cfg["search"]; k, mode, pool = int(k or s["k"]), mode or s["mode"], int(pool or s["pool"])
        if not q or not q.strip(): return []
        if mode == "bm25": idx = self.bm25(q, pool if doc else k)
        elif mode == "dense": idx = self.dense_rank(q, pool if doc else k)
        else:
            score = {}
            for lst in (self.bm25(q, pool), self.dense_rank(q, pool)):
                for r, i in enumerate(lst): score[i] = score.get(i, 0) + 1 / (60 + r)
            idx = sorted(score, key=lambda i: (-score[i], i))
        hits = [self.chunks[i] for i in idx]
        if doc: hits = [c for c in hits if c["doc"] == doc]
        return hits[:k]

    def documents(self):
        out = {}
        for c in self.chunks:
            d = out.setdefault(c["doc"], {"doc": c["doc"], "chunks": 0, "pages": 0})
            d["chunks"] += 1; d["pages"] = max(d["pages"], c.get("page") or 0)
        return list(out.values())
