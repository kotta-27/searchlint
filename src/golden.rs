//! golden.yaml: the queries a retriever must get right.
//!
//! - q: 納税証明書の交付請求の手順
//!   expect: [納税証明書の交付請求]   # a hit = a retrieved chunk that contains ANY of these (whitespace-insensitive)
//!   doc: manual.pdf                 # optional: the chunk must come from this document
//!   status: ok                      # draft rows are skipped by `run`
use crate::config::Config;
use crate::ingest::{norm, Chunk};
use crate::ollama::Ollama;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const MIN_EXPECT: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub q: String,
    #[serde(default, deserialize_with = "one_or_many")]
    pub expect: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src: Option<String>,
    #[serde(default = "ok")]
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}
fn ok() -> String { "ok".into() }

fn one_or_many<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum V { Many(Vec<serde_yaml::Value>), One(serde_yaml::Value) }   // Many first: a bare Value would swallow the list
    let to_s = |v: serde_yaml::Value| match v { serde_yaml::Value::String(s) => s, other => serde_yaml::to_string(&other).unwrap_or_default().trim().to_string() };
    Ok(match V::deserialize(d)? { V::Many(vs) => vs.into_iter().map(to_s).collect(), V::One(v) => vec![to_s(v)] })
}

pub fn load(path: &Path) -> Result<Vec<Row>> {
    if !path.exists() { return Ok(vec![]); }
    let text = std::fs::read_to_string(path)?;
    let raw: serde_yaml::Value = serde_yaml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let Some(items) = raw.as_sequence() else { bail!("{}: top level must be a list", path.display()) };
    let mut rows = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let r: Row = serde_yaml::from_value(item.clone()).map_err(|_| anyhow::anyhow!("{}: entry {} needs `q` and `expect` (got {})", path.display(), i + 1, short(item)))?;
        let r = Row { expect: r.expect.into_iter().filter(|x| !x.trim().is_empty()).collect(), ..r };
        if r.q.trim().is_empty() || r.expect.is_empty() { bail!("{}: entry {} needs `q` and `expect`", path.display(), i + 1); }
        rows.push(r);
    }
    Ok(rows)
}
fn short(v: &serde_yaml::Value) -> String { let s = serde_yaml::to_string(v).unwrap_or_default(); s.trim().chars().take(60).collect() }

pub fn save(path: &Path, rows: &[Row]) -> Result<()> {
    let body = serde_yaml::to_string(rows)?;
    std::fs::write(path, format!("# searchlint golden queries. expect = snippets that must appear in a retrieved chunk. Review drafts, then `searchlint golden approve`.\n{body}"))?;
    Ok(())
}

pub fn warnings(rows: &[Row]) -> Vec<String> {
    rows.iter().flat_map(|r| r.expect.iter().filter(|x| norm(x).chars().count() < MIN_EXPECT).map(move |x| format!("expect too short ({x:?}) in: {}", r.q.chars().take(40).collect::<String>()))).collect()
}

pub fn hit(chunk: &Chunk, row: &Row) -> bool {
    if let Some(d) = &row.doc { if &chunk.doc != d { return false; } }
    let t = norm(&chunk.text);
    row.expect.iter().any(|x| { let n = norm(x); !n.is_empty() && t.contains(&n) })
}

const DRAFT: &str = "次の資料だけから答えられる質問を1つ作ってください。資料固有の事実（固有名詞・数値・手順など）を問う短い質問にします。\n\
answer は資料中にそのまま出てくる語句（4〜30文字）にしてください。\n# 資料\n{text}\nJSON のみ: {\"question\": \"質問文\", \"answer\": \"資料中の語句\"}";

/// LLM drafts n queries; each is kept only if its answer snippet really occurs in the chunk.
pub fn draft(cfg: &Config, chunks: &[Chunk], n: usize, existing: &[Row], seed: u64) -> Result<Vec<Row>> {
    let ollama = Ollama::new(&cfg.llm.host);
    let done: std::collections::HashSet<&str> = existing.iter().filter_map(|r| r.src.as_deref()).collect();
    let mut cands: Vec<&Chunk> = chunks.iter().filter(|c| !done.contains(c.id.as_str()) && c.text.chars().count() > 120).collect();
    // deterministic shuffle (xorshift) so --seed reproduces the draft
    let mut s = seed.max(1);
    for i in (1..cands.len()).rev() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; cands.swap(i, (s % (i as u64 + 1)) as usize); }
    let mut out = Vec::new();
    for c in cands {
        if out.len() >= n { break; }
        let text: String = c.text.chars().take(1500).collect();
        let Some(j) = ollama.chat_json(&cfg.llm.model, &DRAFT.replace("{text}", &text), 200)? else { continue };
        let (Some(q), Some(a)) = (j["question"].as_str(), j["answer"].as_str()) else { continue };
        let a = a.trim().to_string();
        let an = norm(&a);
        if !(MIN_EXPECT..=60).contains(&an.chars().count()) || !norm(&c.text).contains(&an) { continue; }
        out.push(Row { q: q.trim().to_string(), expect: vec![a], doc: Some(c.doc.clone()), src: Some(c.id.clone()), status: "draft".into(), note: None });
    }
    Ok(out)
}
