//! `golden check`: grade the golden set itself, so a recall number can be trusted (or not) before running it.
use crate::config::Config;
use crate::golden::{self, Row};
use crate::index::Index;
use crate::ingest::norm;
use std::collections::{HashMap, HashSet};

pub struct Report { pub errors: usize, pub warnings: usize }

pub fn check(cfg: &Config, ix: &Index, rows: &[Row], out: &mut dyn FnMut(String)) -> anyhow::Result<Report> {
    let hashes = crate::ingest::doc_hashes(cfg)?;
    let live: Vec<&Row> = rows.iter().filter(|r| r.status != "draft").collect();
    let mut problems: Vec<(String, &'static str, String)> = Vec::new(); // (question, level, message)
    let mut overlaps: Vec<f64> = Vec::new();
    for r in &live {
        let n = golden::spread(r, &ix.chunks);
        if n == 0 { problems.push((r.q.clone(), "error", "expect is in no chunk (stale): fix or retire it".into())); continue; }
        if n > golden::MAX_SPREAD { problems.push((r.q.clone(), "warn", format!("expect matches {n} sections: a hit may be the wrong place; use a longer, more specific snippet"))); }
        for e in &r.expect { if norm(e).chars().count() < golden::MIN_EXPECT { problems.push((r.q.clone(), "warn", format!("expect {e:?} is shorter than {} chars", golden::MIN_EXPECT))); } }
        if let Some(why) = golden::vague_question(&r.q) { problems.push((r.q.clone(), "warn", format!("question {why}"))); }
        if golden::state(r, &hashes) == golden::State::Unpinned { problems.push((r.q.clone(), "info", "not pinned to a document version (`searchlint golden pin`)".into())); }
        // how much the question copies the answer chunk's words: high = easy for keyword search, flatters recall
        if let Some(c) = ix.chunks.iter().find(|c| golden::hit(c, r)) {
            let qt: HashSet<String> = crate::tokenize::tokens(&r.q).unwrap_or_default().into_iter().filter(|t| t.chars().count() >= 2).collect();
            let ct: HashSet<String> = crate::tokenize::tokens(&c.text).unwrap_or_default().into_iter().collect();
            if !qt.is_empty() { overlaps.push(qt.intersection(&ct).count() as f64 / qt.len() as f64); }
        }
    }
    // duplicates
    let mut seen: HashMap<String, usize> = HashMap::new();
    for r in &live { *seen.entry(norm(&r.q)).or_default() += 1; }
    for (q, n) in seen.iter().filter(|(_, n)| **n > 1) { problems.push((q.clone(), "warn", format!("question appears {n} times"))); }
    // coverage
    let per_doc: HashMap<&str, usize> = live.iter().filter_map(|r| r.doc.as_deref()).fold(HashMap::new(), |mut m, d| { *m.entry(d).or_default() += 1; m });
    let docs: Vec<&String> = hashes.keys().collect();
    let uncovered: Vec<&&String> = docs.iter().filter(|d| !per_doc.contains_key(d.as_str())).collect();

    let errors = problems.iter().filter(|p| p.1 == "error").count();
    let warnings = problems.iter().filter(|p| p.1 == "warn").count();
    overlaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if overlaps.is_empty() { 0.0 } else { overlaps[overlaps.len() / 2] };
    out(format!("golden check  rows={}  drafts={}  documents={}", live.len(), rows.len() - live.len(), docs.len()));
    out(format!("  errors {errors}   warnings {warnings}"));
    out(format!("  coverage   {}/{} documents have golden rows{}", docs.len() - uncovered.len(), docs.len(),
        if uncovered.is_empty() { String::new() } else { format!(" (uncovered: {}{})", uncovered.iter().take(3).map(|d| d.as_str()).collect::<Vec<_>>().join(", "), if uncovered.len() > 3 { ", ..." } else { "" }) }));
    out(format!("  wording    questions share {:.0}% of their words with the answer chunk (median){}", median * 100.0,
        if median > 0.7 { ": high, keyword search finds these easily and recall will look better than real use" } else { "" }));
    if live.len() < 40 { out(format!("  size       {} rows: one miss moves recall by {:.2}; aim for 40+", live.len(), 1.0 / live.len().max(1) as f64)); }
    let mut by_level: Vec<&(String, &str, String)> = problems.iter().filter(|p| p.1 != "info").collect();
    by_level.sort_by_key(|p| if p.1 == "error" { 0 } else { 1 });
    for (q, level, msg) in by_level.iter().take(20) {
        out(format!("  {:5}  {}   {}", level.to_uppercase(), q.lines().next().unwrap_or("").chars().take(48).collect::<String>(), msg));
    }
    if by_level.len() > 20 { out(format!("  ... {} more", by_level.len() - 20)); }
    let unpinned = problems.iter().filter(|p| p.1 == "info").count();
    if unpinned > 0 { out(format!("  info   {unpinned} rows are not pinned to a document version (`searchlint golden pin`)")); }
    Ok(Report { errors, warnings })
}
