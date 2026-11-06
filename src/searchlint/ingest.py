"""Documents -> cleaned chunks. txt / md / pdf (per page, with page numbers).
Cleaning removes running headers/footers, page numbers, isolated step callouts and TOC dot leaders."""
import os, re, glob, hashlib
from .llm import sha

DOTS = re.compile(r"[.．…・]{5,}")
NUMLINE = re.compile(r"^\s*[0-9０-９]{1,3}\s*(/\s*[0-9０-９]{1,3})?\s*$|^\s*[①-⑳]\s*$")

def extract(path):
    """-> [(page_or_None, text)]"""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages)]
    return [(None, open(path, encoding="utf-8", errors="replace").read())]

def clean_pages(pages):
    if len(pages) >= 5:
        freq = {}
        for _, body in pages:
            for ln in {l.strip() for l in body.split("\n") if l.strip()}: freq[ln] = freq.get(ln, 0) + 1
        boiler = {ln for ln, n in freq.items() if n >= 0.3 * len(pages) and len(ln) <= 80}
    else:
        boiler = set()
    out = []
    for pg, body in pages:
        keep = [ln for ln in body.split("\n") if ln.strip() and ln.strip() not in boiler and not NUMLINE.match(ln.strip()) and not DOTS.search(ln)]
        out.append((pg, "\n".join(keep)))
    return out

def split(text, size, overlap):
    pieces = [p for p in re.split(r"(?<=[。．！？\n])", text) if p.strip()]
    chunks, cur = [], ""
    for p in pieces:
        if len(cur) + len(p) > size and cur:
            chunks.append(cur.strip()); cur = cur[-overlap:] if overlap else ""
        cur += p
    if cur.strip(): chunks.append(cur.strip())
    return chunks

def list_files(docs_dir, patterns):
    files = set()
    for pat in patterns:
        files.update(glob.glob(os.path.join(docs_dir, pat), recursive=True))
    return sorted(f for f in files if os.path.isfile(f))

def docs_manifest(cfg):
    """hash of (relative path, size, mtime) for every included file: tells `run` whether the index is stale"""
    files = list_files(cfg.docs, cfg["include"])
    h = hashlib.sha256()
    for f in files:
        st = os.stat(f); h.update(f"{os.path.relpath(f, cfg.docs)}|{st.st_size}|{int(st.st_mtime)}\n".encode())
    return h.hexdigest()[:10], files

def build_chunks(cfg):
    """-> (files, chunks). chunk = {id, doc, page, text, sha}; id = doc#n. Duplicates are dropped within a document only,
    so a boilerplate paragraph shared by two documents stays searchable in both."""
    _, files = docs_manifest(cfg)
    size, overlap, clean = int(cfg["chunk"]["size"]), int(cfg["chunk"]["overlap"]), cfg["chunk"].get("clean", True)
    chunks = []
    for f in files:
        rel = os.path.relpath(f, cfg.docs)
        pages = extract(f)
        if clean: pages = clean_pages(pages)
        n, seen = 0, set()
        for page, body in pages:
            for t in split(body, size, overlap):
                key = re.sub(r"\s+", "", t)
                if key in seen: continue
                seen.add(key)
                chunks.append({"id": f"{rel}#{n}", "doc": rel, "page": page, "text": t, "sha": sha(t)}); n += 1
    return files, chunks
