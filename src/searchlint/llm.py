"""Ollama clients: embeddings and chat. Only the golden-draft step uses chat."""
import json, hashlib, httpx, numpy as np

def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

class Ollama:
    def __init__(self, host="http://localhost:11434"):
        self.c = httpx.Client(base_url=host, timeout=httpx.Timeout(connect=10, read=300, write=60, pool=30))

    def embed(self, model: str, texts: list[str]) -> np.ndarray:
        r = self.c.post("/api/embed", json={"model": model, "input": texts}); r.raise_for_status()
        v = np.array(r.json()["embeddings"], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def chat(self, model: str, prompt: str, json_mode=False, max_tokens=300) -> str:
        body = {"model": model, "messages": [{"role": "user", "content": prompt + ("\n/no_think" if "qwen3" in model else "")}],
                "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": max_tokens}}
        if json_mode: body["format"] = "json"
        r = self.c.post("/api/chat", json=body); r.raise_for_status()
        out = r.json()["message"]["content"]
        if "</think>" in out: out = out.split("</think>")[-1]
        return out.strip()

    def chat_json(self, model, prompt, max_tokens=300):
        try: return json.loads(self.chat(model, prompt, json_mode=True, max_tokens=max_tokens))
        except Exception: return None
