"""searchlint.yaml: one file per repo describing the corpus, the index and the gate."""
import os, json, hashlib, yaml

DEFAULT = {
    "docs": "./docs",
    "include": ["**/*.md", "**/*.txt", "**/*.pdf"],
    "chunk": {"size": 500, "overlap": 60, "clean": True},
    "embed": {"provider": "ollama", "model": "bge-m3", "host": "http://localhost:11434"},
    "llm": {"provider": "ollama", "model": "qwen3:8b", "host": "http://localhost:11434"},
    "search": {"mode": "hybrid", "k": 5, "pool": 30},
    "gate": {"recall": 0.9, "mrr": 0.0, "max_drop": 0.02},
    "golden": "golden.yaml",
    "baseline": "searchlint.baseline.json",   # commit this: the reference run the CI gate compares against
    "state": ".searchlint",                   # ignore this: index, embedding cache, last run, query log
}

class ConfigError(Exception): pass

def deep_merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out

class Config:
    def __init__(self, path="searchlint.yaml"):
        self.path = os.path.abspath(path)
        self.root = os.path.dirname(self.path)
        user = yaml.safe_load(open(self.path)) if os.path.exists(self.path) else {}
        self.d = deep_merge(DEFAULT, user or {})
        self.validate()

    def validate(self):
        c, s = self.d["chunk"], self.d["search"]
        if not (50 <= int(c["size"]) <= 5000): raise ConfigError("chunk.size must be 50..5000")
        if not (0 <= int(c["overlap"]) < int(c["size"]) / 2): raise ConfigError("chunk.overlap must be < chunk.size / 2")
        if s["mode"] not in ("bm25", "dense", "hybrid"): raise ConfigError("search.mode must be bm25 | dense | hybrid")
        if not (1 <= int(s["k"]) <= 100): raise ConfigError("search.k must be 1..100")

    def __getitem__(self, k): return self.d[k]
    def abs(self, p): return p if os.path.isabs(p) else os.path.join(self.root, p)
    @property
    def docs(self): return self.abs(self.d["docs"])
    @property
    def state(self): return self.abs(self.d["state"])
    @property
    def golden_path(self): return self.abs(self.d["golden"])
    @property
    def baseline_path(self): return self.abs(self.d["baseline"])
    @property
    def needs_embeddings(self): return self.d["search"]["mode"] != "bm25"

    def fingerprint(self):
        """what the index and the baseline depend on: chunking, embedding model, search settings"""
        key = {"chunk": self.d["chunk"], "embed": self.d["embed"]["model"] if self.needs_embeddings else None, "search": self.d["search"]}
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]

    @staticmethod
    def write_default(path="searchlint.yaml", docs="./docs"):
        d = dict(DEFAULT); d["docs"] = docs
        with open(path, "w") as f:
            f.write("# searchlint: retrieval lint config. Commit this file, golden.yaml and searchlint.baseline.json; ignore .searchlint/\n")
            yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)
