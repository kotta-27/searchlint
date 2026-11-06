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
    /// content hash of `doc` when this row was last confirmed (`golden pin` / `golden approve`).
    /// If the document changes, the row is not gated any more: it goes to review until re-pinned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_sha: Option<String>,
    /// the text around the answer when pinned: used to find where the answer moved after the document changed
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}
fn ok() -> String { "ok".into() }

/// How a row relates to the current documents.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum State { Unchanged, Unpinned, Changed, Removed }

pub fn state(r: &Row, hashes: &std::collections::HashMap<String, String>) -> State {
    match (&r.doc, &r.doc_sha) {
        (Some(d), Some(h)) => match hashes.get(d) { None => State::Removed, Some(cur) if cur == h => State::Unchanged, Some(_) => State::Changed },
        _ => State::Unpinned,
    }
}

/// Record doc, doc_sha and anchor for every row whose expect is found in the corpus now. Rows whose expect is gone keep
/// their old pin (so they stay in review). Returns (pinned, not found).
pub fn pin(rows: &mut [Row], chunks: &[Chunk], hashes: &std::collections::HashMap<String, String>) -> (usize, usize) {
    let (mut pinned, mut missing) = (0, 0);
    for r in rows.iter_mut().filter(|r| r.status != "draft") {
        match chunks.iter().find(|c| hit(c, r)) {
            Some(c) => {
                r.doc = Some(c.doc.clone());
                r.doc_sha = hashes.get(&c.doc).cloned();
                r.anchor = Some(anchor_around(&c.text, &r.expect));
                pinned += 1;
            }
            None => missing += 1,
        }
    }
    (pinned, missing)
}

/// the answer with ~60 chars of context on each side (whitespace removed): enough to find where it went after an edit
fn anchor_around(text: &str, expect: &[String]) -> String {
    let t: Vec<char> = norm(text).chars().collect();
    let ts: String = t.iter().collect();
    for e in expect {
        let en = norm(e);
        if let Some(bpos) = ts.find(&en) {
            let start = ts[..bpos].chars().count();
            let end = start + en.chars().count();
            let (mut a, mut b) = (start.saturating_sub(60), (end + 60).min(t.len()));
            // keep the anchor inside the answer's own section: an edit to the neighbouring section must not move this row to review
            if let Some(h) = (a..start).rev().find(|&i| t[i] == '#') { a = h + 1; while a < start && t[a] == '#' { a += 1; } }
            if let Some(h) = (end..b).find(|&i| t[i] == '#') { b = h; }
            return t[a..b].iter().collect();
        }
    }
    t.iter().take(300).collect()
}

#[allow(dead_code)]
fn trigrams(s: &str) -> std::collections::HashSet<[char; 3]> {
    let v: Vec<char> = norm(s).chars().collect();
    v.windows(3).map(|w| [w[0], w[1], w[2]]).collect()
}

#[allow(dead_code)]
fn longest_common(a: &str, b: &str) -> String {
    let (a, b): (Vec<char>, Vec<char>) = (norm(a).chars().collect(), norm(b).chars().collect());
    let mut prev = vec![0usize; b.len() + 1];
    let (mut best, mut end) = (0usize, 0usize);
    for i in 1..=a.len() {
        let mut cur = vec![0usize; b.len() + 1];
        for j in 1..=b.len() {
            if a[i - 1] == b[j - 1] { cur[j] = prev[j - 1] + 1; if cur[j] > best { best = cur[j]; end = i; } }
        }
        prev = cur;
    }
    a[end - best..end].iter().collect()
}

pub enum Moved<'a> {
    /// what now sits between the old answer's left and right context
    Suggest(&'a Chunk, String),
    /// the surrounding text is partly there but the answer cannot be pinned down: rewrite by hand from this chunk
    Near(&'a Chunk),
    /// neither side of the old context survives in the document: the section was removed or rewritten
    Gone,
}

/// For a row whose answer context changed: where it went, judged from the text around the old answer.
pub fn relocate<'a>(r: &Row, chunks: &'a [Chunk]) -> Moved<'a> {
    let (Some(anchor), Some(doc)) = (r.anchor.as_deref(), r.doc.as_deref()) else { return Moved::Gone };
    let doc_chunks: Vec<&Chunk> = chunks.iter().filter(|c| c.doc == doc).collect();
    let a: Vec<char> = anchor.chars().collect();
    let Some(old) = r.expect.iter().map(|e| norm(e)).find(|e| anchor.contains(e.as_str())) else { return Moved::Gone };
    let start = anchor[..anchor.find(&old).unwrap()].chars().count();
    let end = start + old.chars().count();
    let mut side_seen = None;
    for w in [16usize, 12, 8, 6] {
        let left: String = a[start.saturating_sub(w)..start].iter().collect();
        let right: String = a[end..(end + w).min(a.len())].iter().collect();
        for c in &doc_chunks {
            let t = norm(&c.text);
            let li = if left.is_empty() { None } else { t.find(&left) };
            let ri = if right.is_empty() { None } else { t.find(&right) };
            if li.is_some() || ri.is_some() { side_seen.get_or_insert(*c); }
            if let Some(li) = li {
                let from = li + left.len();
                if let Some(rj) = (!right.is_empty()).then(|| t[from..].find(&right)).flatten() {
                    let mid = t[from..from + rj].to_string();
                    let n = mid.chars().count();
                    if (2..=60).contains(&n) && mid != old { return Moved::Suggest(c, mid); }
                }
            }
        }
    }
    match side_seen { Some(c) => Moved::Near(c), None => Moved::Gone }
}

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

/// answers that are headings, page / section references, or just the question's own words make useless golden rows
fn bad_answer(q: &str, a: &str) -> bool {
    let an = norm(a);
    let re = regex::Regex::new(r"^[0-9０-９]+([-\.．－][0-9０-９]+)*\s*|ページ|^第[0-9０-９一二三四五六七八九十]+[章節条]|[～〜]").unwrap();
    re.is_match(a.trim()) || norm(q).contains(&an) || a.trim().ends_with("する") && a.chars().count() <= 8
}

/// LLM drafts n queries; each is kept only if its answer snippet really occurs in the chunk and is not a heading / reference.
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
        if !(MIN_EXPECT..=60).contains(&an.chars().count()) || !norm(&c.text).contains(&an) || bad_answer(q, &a) { continue; }
        out.push(Row { q: q.trim().to_string(), expect: vec![a], doc: Some(c.doc.clone()), src: Some(c.id.clone()), status: "draft".into(), note: None, doc_sha: None, anchor: None });
    }
    Ok(out)
}
