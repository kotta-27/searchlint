"""searchlint.yaml: one file per repo describing the corpus, the index and the gate."""
import os, yaml

DEFAULT = {
    "docs": "./docs",
    "include": ["**/*.md", "**/*.txt", "**/*.pdf"],
    "chunk": {"size": 500, "overlap": 60, "clean": True},
    "embed": {"provider": "ollama", "model": "bge-m3", "host": "http://localhost:11434"},
    "llm": {"provider": "ollama", "model": "qwen3:8b", "host": "http://localhost:11434"},
    "search": {"mode": "hybrid", "k": 5, "pool": 30},
    "gate": {"recall": 0.9, "mrr": 0.0, "max_drop": 0.02},
    "golden": "golden.yaml",
    "state": ".searchlint",
}

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

    def __getitem__(self, k): return self.d[k]
    def abs(self, p): return p if os.path.isabs(p) else os.path.join(self.root, p)
    @property
    def docs(self): return self.abs(self.d["docs"])
    @property
    def state(self): return self.abs(self.d["state"])
    @property
    def golden_path(self): return self.abs(self.d["golden"])

    @staticmethod
    def write_default(path="searchlint.yaml", docs="./docs"):
        d = dict(DEFAULT); d["docs"] = docs
        with open(path, "w") as f:
            f.write("# searchlint: retrieval lint config\n")
            yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)
