//! Evaluate the index against golden.yaml: recall@k and MRR per query; compare with the committed baseline; gate.
use crate::config::Config;
use crate::golden::{self, Row};
use crate::index::Index;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Top { pub id: String, pub doc: String, pub page: Option<u32>, pub text: String }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Per {
    pub q: String, pub hit: bool, pub rank: Option<usize>, pub mrr: f64, pub top: Vec<Top>,
    /// for misses: "stale" (no chunk in the corpus contains the expected snippet: update golden) or "rank N" / "beyond pool" (retrieval)
    #[serde(default, skip_serializing_if = "Option::is_none")] pub why: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Summary {
    pub n: usize, pub recall: f64, pub mrr: f64,
    /// search latency per query (ms), measured after one warm-up query; the same search path the MCP server uses
    #[serde(default)] pub p50_ms: f64, #[serde(default)] pub p95_ms: f64, #[serde(default)] pub max_ms: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub at: String, pub k: usize, pub mode: String, pub fingerprint: String, pub index_built: String,
    pub summary: Summary, pub per: Vec<Per>,
    #[serde(default)] pub problems: Vec<String>, #[serde(default)] pub notes: Vec<String>, #[serde(default)] pub ok: bool,
    /// rows whose document changed since they were pinned: evaluated but not gated, with a suggestion each
    #[serde(default)] pub review: Vec<serde_json::Value>,
    /// rows whose document is gone
    #[serde(default)] pub removed: Vec<String>,
    /// documents no golden row points at
    #[serde(default)] pub uncovered: Vec<String>,
}

pub fn evaluate(index: &Index, rows: &[Row], k: usize) -> Result<(Summary, Vec<Per>)> {
    let mut per = Vec::new();
    let mut lat: Vec<f64> = Vec::new();
    if let Some(r) = rows.first() { index.search(&r.q, Some(k), None)?; } // warm-up (first inference allocates)
    for r in rows {
        let t0 = std::time::Instant::now();
        let chunks = index.search(&r.q, Some(k), None)?;
        lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        let rank = chunks.iter().position(|c| golden::hit(c, r));
        let why = if rank.is_some() { None } else if !index.chunks.iter().any(|c| golden::hit(c, r)) {
            Some("stale".into())
        } else {
            let pool = index.cfg.search.pool.max(k);
            match index.search(&r.q, Some(pool), None)?.iter().position(|c| golden::hit(c, r)) { Some(i) => Some(format!("rank {}", i + 1)), None => Some(format!("beyond pool {pool}")) }
        };
        per.push(Per {
            q: r.q.clone(), hit: rank.is_some(), rank: rank.map(|i| i + 1), mrr: rank.map(|i| 1.0 / (i as f64 + 1.0)).unwrap_or(0.0),
            top: chunks.iter().take(3).map(|c| Top { id: c.id.clone(), doc: c.doc.clone(), page: c.page, text: c.text.chars().take(120).collect() }).collect(),
            why,
        });
    }
    let n = per.len().max(1) as f64;
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |q: f64| if lat.is_empty() { 0.0 } else { lat[((lat.len() as f64 * q).ceil() as usize).saturating_sub(1).min(lat.len() - 1)] };
    let round = |x: f64| (x * 10.0).round() / 10.0;
    Ok((Summary { n: per.len(), recall: per.iter().filter(|p| p.hit).count() as f64 / n, mrr: per.iter().map(|p| p.mrr).sum::<f64>() / n,
                  p50_ms: round(pct(0.5)), p95_ms: round(pct(0.95)), max_ms: round(lat.last().copied().unwrap_or(0.0)) }, per))
}

pub struct Opts { pub ci: bool, pub update_baseline: bool }

/// first line of a question, cut to n chars, for one-line listings
fn short(q: &str, n: usize) -> String { q.lines().next().unwrap_or("").chars().take(n).collect() }

pub fn run(cfg: &Config, index: &Index, rows: &[Row], o: &Opts, out: &mut dyn FnMut(String)) -> Result<(bool, RunResult)> {
    let k = cfg.search.k;
    // Only rows whose document is unchanged since they were pinned (or that were never pinned) are gated.
    // A changed document legitimately moves or rewrites answers: those rows go to review instead of failing the build.
    let hashes = crate::ingest::doc_hashes(cfg)?;
    // A changed document only moves the rows whose own answer context changed: a row whose anchor (the answer with
    // ~60 chars around it) is still verbatim in the new text stays gated.
    let (mut gated, mut changed, mut removed) = (Vec::new(), Vec::new(), Vec::new());
    let mut texts: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut kept = 0usize;
    for r in rows {
        match golden::state(r, &hashes) {
            golden::State::Unchanged | golden::State::Unpinned => gated.push(r.clone()),
            golden::State::Changed => {
                let (d, a) = (r.doc.clone().unwrap_or_default(), r.anchor.clone().unwrap_or_default());
                if !texts.contains_key(&d) { texts.insert(d.clone(), crate::ingest::doc_text(cfg, &d).unwrap_or_default()); }
                if !a.is_empty() && texts[&d].contains(&a) { kept += 1; gated.push(r.clone()); } else { changed.push(r.clone()); }
            }
            golden::State::Removed => removed.push(r.clone()),
        }
    }
    let unpinned = rows.iter().filter(|r| golden::state(r, &hashes) == golden::State::Unpinned).count();
    let (summary, per) = evaluate(index, &gated, k)?;
    let (_, per_changed) = evaluate(index, &changed, k)?;
    let mut review = Vec::new();
    for (r, p) in changed.iter().zip(&per_changed) {
        let v = if p.hit {
            serde_json::json!({"q": r.q, "doc": r.doc, "status": "still found", "action": "re-pin (searchlint golden pin)", "rank": p.rank})
        } else if p.why.as_deref() == Some("stale") {
            match golden::relocate(r, &index.chunks) {
                golden::Moved::Suggest(c, sugg) => serde_json::json!({"q": r.q, "doc": r.doc, "status": "answer text changed", "action": "update expect", "suggest": sugg, "near": c.id, "page": c.page}),
                golden::Moved::Near(c) => serde_json::json!({"q": r.q, "doc": r.doc, "status": "answer text changed", "action": "rewrite expect by hand from the nearest chunk", "near": c.id, "page": c.page}),
                golden::Moved::Gone => serde_json::json!({"q": r.q, "doc": r.doc, "status": "section gone", "action": "retire or rewrite"}),
            }
        } else {
            serde_json::json!({"q": r.q, "doc": r.doc, "status": "answer still present but not retrieved", "action": "retrieval regression in a changed document", "why": p.why})
        };
        review.push(v);
    }
    let covered: std::collections::HashSet<&str> = rows.iter().filter_map(|r| r.doc.as_deref()).collect();
    let mut uncovered: Vec<String> = if covered.is_empty() { vec![] } else { hashes.keys().filter(|d| !covered.contains(d.as_str())).cloned().collect() };
    uncovered.sort();
    let mut result = RunResult {
        at: chrono::Local::now().format("%Y-%m-%dT%H:%M").to_string(), k, mode: cfg.search.mode.clone(), fingerprint: cfg.fingerprint(),
        index_built: index.meta.built.clone(), summary: summary.clone(), per: per.clone(), problems: vec![], notes: vec![], ok: false,
        review: review.clone(), removed: removed.iter().map(|r| r.q.clone()).collect(), uncovered: uncovered.clone(),
    };
    fs::create_dir_all(cfg.state_dir())?;
    fs::write(cfg.state_dir().join("last.json"), serde_json::to_string_pretty(&result)?)?;

    let base: Option<RunResult> = fs::read_to_string(cfg.baseline_path()).ok().and_then(|s| serde_json::from_str(&s).ok());
    let comparable = base.as_ref().map(|b| b.fingerprint == result.fingerprint).unwrap_or(false);
    let gate = &cfg.gate;
    let (mut problems, mut notes) = (Vec::new(), Vec::new());

    out(format!("searchlint  n={}  k={k}  mode={}{}", summary.n, cfg.search.mode,
        if changed.is_empty() && removed.is_empty() { String::new() } else { format!("   ({} rows in review: their documents changed or were removed)", changed.len() + removed.len()) }));
    let line = |name: &str, v: f64, bv: Option<f64>| -> String {
        let mut s = format!("  {name:10} {v:.3}");
        if let Some(b) = bv { s += &format!("   (baseline {b:.3}, {:+.3})", v - b); }
        s
    };
    out(line("recall", summary.recall, comparable.then(|| base.as_ref().unwrap().summary.recall)));
    out(line("mrr", summary.mrr, comparable.then(|| base.as_ref().unwrap().summary.mrr)));
    {
        let mut l = format!("  latency    p50 {:.0} ms  p95 {:.0} ms  max {:.0} ms", summary.p50_ms, summary.p95_ms, summary.max_ms);
        if let Some(b) = base.as_ref().filter(|b| comparable && b.summary.p95_ms > 0.0) { l += &format!("   (baseline p95 {:.0} ms)", b.summary.p95_ms); }
        out(l);
    }
    for w in golden::warnings(rows).into_iter().take(5) { notes.push(format!("warning: {w}")); }
    let misses: Vec<&Per> = per.iter().filter(|p| !p.hit).collect();
    if !misses.is_empty() {
        let stale = misses.iter().filter(|p| p.why.as_deref() == Some("stale")).count();
        out(format!("  missed {}{}:", misses.len(), if stale > 0 { format!(" ({stale} stale golden: the expected snippet is in no chunk any more; update golden.yaml)") } else { String::new() }));
        for p in misses.iter().take(10) {
            let t = p.top.first().map(|t| format!("{}{}", t.doc, t.page.map(|pg| format!(" p.{pg}")).unwrap_or_default())).unwrap_or("-".into());
            let why = match p.why.as_deref() { Some("stale") => "STALE GOLDEN".to_string(), Some(w) => format!("retrieval: {w}"), None => String::new() };
            out(format!("    - {}   [{why}]  top1: {t}", short(&p.q, 60)));
        }
    }
    if let Some(b) = &base {
        if !comparable {
            notes.push("baseline was made with a different config (chunk / embed / search): drop check skipped. Run `searchlint run --baseline` to reset it.".into());
        } else {
            let was: std::collections::HashMap<&str, &Per> = b.per.iter().map(|p| (p.q.as_str(), p)).collect();
            // compare on the rows gated now that the baseline also had (rows moved to review are not held against the run)
            let both: Vec<&Per> = per.iter().filter(|p| was.contains_key(p.q.as_str())).collect();
            let now_rate = both.iter().filter(|p| p.hit).count() as f64 / both.len().max(1) as f64;
            let base_rate = both.iter().filter(|p| was[p.q.as_str()].hit).count() as f64 / both.len().max(1) as f64;
            let broken: Vec<&Per> = per.iter().filter(|p| !p.hit && was.get(p.q.as_str()).map(|w| w.hit).unwrap_or(false)).collect();
            // still found but ranked lower than in the baseline: the early sign of crowding (a new or duplicated document
            // outranking the right one). Name the document that now sits on top.
            let sank: Vec<(&Per, usize, usize)> = per.iter().filter_map(|p| {
                let w = was.get(p.q.as_str())?;
                match (w.rank, p.rank) { (Some(a), Some(b)) if b > a => Some((p, a, b)), _ => None }
            }).collect();
            if !sank.is_empty() {
                out(format!("  ranked lower than baseline {}:", sank.len()));
                for (p, a, b) in sank.iter().take(8) {
                    let t = p.top.first().map(|t| t.doc.clone()).unwrap_or_default();
                    out(format!("    - {}   rank {a} -> {b}   top1 now: {t}", short(&p.q, 50)));
                }
                let mut by_doc: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
                for (p, _, _) in &sank { if let Some(t) = p.top.first() { *by_doc.entry(t.doc.as_str()).or_default() += 1; } }
                if let Some((d, n)) = by_doc.into_iter().max_by_key(|x| x.1).filter(|x| x.1 >= 2) {
                    notes.push(format!("{d} is now the top result for {n} of the rows that ranked lower: a new or duplicated document may be crowding out the right one"));
                }
            }
            let fixed = per.iter().filter(|p| p.hit && was.get(p.q.as_str()).map(|w| !w.hit).unwrap_or(false)).count();
            if fixed > 0 || !broken.is_empty() {
                out(format!("  vs baseline: {fixed} fixed, {} broken{}", broken.len(), if broken.is_empty() { String::new() } else { format!(": {}", broken.iter().take(5).map(|p| short(&p.q, 40)).collect::<Vec<_>>().join("; ")) }));
            }
            let now_mrr = both.iter().map(|p| p.mrr).sum::<f64>() / both.len().max(1) as f64;
            let base_mrr = both.iter().map(|p| was[p.q.as_str()].mrr).sum::<f64>() / both.len().max(1) as f64;
            if gate.max_mrr_drop > 0.0 && now_mrr < base_mrr - gate.max_mrr_drop {
                problems.push(format!("mrr dropped {:.3} on {} rows shared with the baseline (> max_mrr_drop {})", base_mrr - now_mrr, both.len(), gate.max_mrr_drop));
            }
            if now_rate < base_rate - gate.max_drop {
                problems.push(format!("recall dropped {:.3} on {} rows shared with the baseline (> max_drop {})", base_rate - now_rate, both.len(), gate.max_drop));
            }
        }
    }
    if kept > 0 { notes.push(format!("{kept} rows' documents changed but their answer context did not: still gated (`searchlint golden pin` refreshes their version)")); }
    if !review.is_empty() {
        out(format!("  review {} (answer context changed since pinned; not gated):", review.len()));
        for v in review.iter().take(10) {
            let extra = v.get("suggest").and_then(|x| x.as_str()).map(|x| format!("  suggest expect: {x:?}")).unwrap_or_default();
            out(format!("    - {}   [{}]{}", short(v["q"].as_str().unwrap_or(""), 50), v["status"].as_str().unwrap_or(""), extra));
        }
    }
    if !removed.is_empty() { out(format!("  removed {} (their document is gone; retire or move them): {}", removed.len(), removed.iter().take(3).map(|r| short(&r.q, 30)).collect::<Vec<_>>().join("; "))); }
    if !uncovered.is_empty() { notes.push(format!("{} document(s) have no golden rows: {}{}. `searchlint golden draft` can start them.", uncovered.len(), uncovered.iter().take(3).cloned().collect::<Vec<_>>().join(", "), if uncovered.len() > 3 { ", ..." } else { "" })); }
    if unpinned > 0 { notes.push(format!("{unpinned} golden rows are not pinned to a document version: run `searchlint golden pin` so document edits move them to review instead of failing.")); }
    if summary.recall < gate.recall { problems.push(format!("recall {:.3} < gate.recall {}", summary.recall, gate.recall)); }
    if summary.mrr < gate.mrr { problems.push(format!("mrr {:.3} < gate.mrr {}", summary.mrr, gate.mrr)); }
    if gate.p95_ms > 0.0 && summary.p95_ms > gate.p95_ms { problems.push(format!("latency p95 {:.0} ms > gate.p95_ms {}", summary.p95_ms, gate.p95_ms)); }
    if summary.n < 20 { notes.push(format!("only {} golden queries: one miss = {:.2} recall. The gate gets meaningful around 40+.", summary.n, 1.0 / summary.n.max(1) as f64)); }
    for n in &notes { out(format!("  {n}")); }
    let ok = problems.is_empty();
    if o.ci {
        for p in &problems { out(format!("  FAIL {p}")); }
        out(if ok { "  PASS".into() } else { "  gate failed".into() });
    }
    result.problems = problems; result.notes = notes; result.ok = ok;
    if o.update_baseline || (base.is_none() && ok) {
        fs::write(cfg.baseline_path(), serde_json::to_string_pretty(&result)?)?;
        out(format!("  baseline saved -> {}{}", cfg.baseline, if o.update_baseline { "" } else { " (first passing run; commit it)" }));
    }
    Ok((ok, result))
}
