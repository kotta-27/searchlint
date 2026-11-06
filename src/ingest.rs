//! Documents -> cleaned chunks. txt / md / pdf (per page, with page numbers).
//! Cleaning removes running headers/footers, page numbers, isolated step callouts and TOC dot leaders.
use crate::config::Config;
use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub doc: String,
    pub page: Option<u32>,
    pub text: String,
    pub sha: String,
}

static DOTS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[.．…・]{5,}").unwrap());
static NUMLINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*[0-9０-９]{1,3}\s*(/\s*[0-9０-９]{1,3})?\s*$|^\s*[①-⑳]\s*$").unwrap());
static WS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

pub fn sha(text: &str) -> String { hex::encode(Sha256::digest(text.as_bytes()))[..16].to_string() }
pub fn norm(s: &str) -> String { WS.replace_all(s, "").to_string() }

/// -> [(page, text)]; page is None for plain text files
pub fn extract(path: &Path) -> Result<Vec<(Option<u32>, String)>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext == "pdf" {
        let bytes = std::fs::read(path)?;
        let pages = pdf_extract::extract_text_from_mem_by_pages(&bytes).with_context(|| format!("pdf {}", path.display()))?;
        return Ok(pages.into_iter().enumerate().map(|(i, t)| (Some(i as u32 + 1), t)).collect());
    }
    Ok(vec![(None, std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?)])
}

pub fn clean_pages(pages: Vec<(Option<u32>, String)>) -> Vec<(Option<u32>, String)> {
    let boiler: HashSet<String> = if pages.len() >= 5 {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for (_, body) in &pages {
            let lines: HashSet<&str> = body.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
            for l in lines { *freq.entry(l.to_string()).or_default() += 1; }
        }
        let min = ((pages.len() as f64) * 0.3).ceil() as usize;
        freq.into_iter().filter(|(l, n)| *n >= min && l.chars().count() <= 80).map(|(l, _)| l).collect()
    } else {
        HashSet::new()
    };
    pages
        .into_iter()
        .map(|(pg, body)| {
            let keep: Vec<&str> = body
                .lines()
                .filter(|l| { let t = l.trim(); !t.is_empty() && !boiler.contains(t) && !NUMLINE.is_match(t) && !DOTS.is_match(l) })
                .collect();
            (pg, keep.join("\n"))
        })
        .collect()
}

/// sentence-ish pieces -> windows of ~size chars with overlap (character counts, not bytes)
pub fn split(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let mut pieces: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '。' | '．' | '！' | '？' | '\n') { pieces.push(std::mem::take(&mut cur)); }
    }
    if !cur.is_empty() { pieces.push(cur); }
    let mut chunks = Vec::new();
    let mut buf = String::new();
    let mut buf_len = 0usize;
    for p in pieces.into_iter().filter(|p| !p.trim().is_empty()) {
        let pl = p.chars().count();
        if buf_len + pl > size && buf_len > 0 {
            chunks.push(buf.trim().to_string());
            let tail: String = if overlap > 0 { buf.chars().rev().take(overlap).collect::<Vec<_>>().into_iter().rev().collect() } else { String::new() };
            buf = tail;
            buf_len = buf.chars().count();
        }
        buf.push_str(&p);
        buf_len += pl;
    }
    if !buf.trim().is_empty() { chunks.push(buf.trim().to_string()); }
    chunks
}

pub fn list_files(cfg: &Config) -> Result<Vec<PathBuf>> {
    let mut out: HashSet<PathBuf> = HashSet::new();
    for pat in &cfg.include {
        let full = cfg.docs_dir().join(pat);
        for entry in glob::glob(&full.to_string_lossy())? { if let Ok(p) = entry { if p.is_file() { out.insert(p); } } }
    }
    let mut v: Vec<PathBuf> = out.into_iter().collect();
    v.sort();
    Ok(v)
}

/// hash of (relative path, size, mtime) for every included file: tells `run` whether the index is stale
pub fn docs_manifest(cfg: &Config) -> Result<(String, Vec<PathBuf>)> {
    let files = list_files(cfg)?;
    let mut h = Sha256::new();
    for f in &files {
        let st = std::fs::metadata(f)?;
        let mtime = st.modified()?.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        h.update(format!("{}|{}|{}\n", rel(cfg, f), st.len(), mtime).as_bytes());
    }
    Ok((hex::encode(h.finalize())[..10].to_string(), files))
}

pub fn rel(cfg: &Config, f: &Path) -> String {
    f.strip_prefix(cfg.docs_dir()).unwrap_or(f).to_string_lossy().replace('\\', "/")
}

/// Duplicates are dropped within a document only, so boilerplate shared by two documents stays searchable in both.
pub fn build_chunks(cfg: &Config) -> Result<(Vec<PathBuf>, Vec<Chunk>)> {
    let (_, files) = docs_manifest(cfg)?;
    let mut chunks = Vec::new();
    for f in &files {
        let doc = rel(cfg, f);
        let mut pages = extract(f)?;
        if cfg.chunk.clean { pages = clean_pages(pages); }
        let mut n = 0usize;
        let mut seen: HashSet<String> = HashSet::new();
        for (page, body) in pages {
            for t in split(&body, cfg.chunk.size, cfg.chunk.overlap) {
                let key = norm(&t);
                if !seen.insert(key) { continue; }
                chunks.push(Chunk { id: format!("{doc}#{n}"), doc: doc.clone(), page, sha: sha(&t), text: t });
                n += 1;
            }
        }
    }
    Ok((files, chunks))
}
