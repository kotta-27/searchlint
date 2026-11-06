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
pub struct Summary { pub n: usize, pub recall: f64, pub mrr: f64 }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub at: String, pub k: usize, pub mode: String, pub fingerprint: String, pub index_built: String,
    pub summary: Summary, pub per: Vec<Per>,
    #[serde(default)] pub problems: Vec<String>, #[serde(default)] pub notes: Vec<String>, #[serde(default)] pub ok: bool,
}

pub fn evaluate(index: &Index, rows: &[Row], k: usize) -> Result<(Summary, Vec<Per>)> {
    let mut per = Vec::new();
    for r in rows {
        let chunks = index.search(&r.q, Some(k), None)?;
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
    Ok((Summary { n: per.len(), recall: per.iter().filter(|p| p.hit).count() as f64 / n, mrr: per.iter().map(|p| p.mrr).sum::<f64>() / n }, per))
}

pub struct Opts { pub ci: bool, pub update_baseline: bool }

pub fn run(cfg: &Config, index: &Index, rows: &[Row], o: &Opts, out: &mut dyn FnMut(String)) -> Result<(bool, RunResult)> {
    let k = cfg.search.k;
    let (summary, per) = evaluate(index, rows, k)?;
    let mut result = RunResult {
        at: chrono::Local::now().format("%Y-%m-%dT%H:%M").to_string(), k, mode: cfg.search.mode.clone(), fingerprint: cfg.fingerprint(),
        index_built: index.meta.built.clone(), summary: summary.clone(), per: per.clone(), problems: vec![], notes: vec![], ok: false,
    };
    fs::create_dir_all(cfg.state_dir())?;
    fs::write(cfg.state_dir().join("last.json"), serde_json::to_string_pretty(&result)?)?;

    let base: Option<RunResult> = fs::read_to_string(cfg.baseline_path()).ok().and_then(|s| serde_json::from_str(&s).ok());
    let comparable = base.as_ref().map(|b| b.fingerprint == result.fingerprint).unwrap_or(false);
    let gate = &cfg.gate;
    let (mut problems, mut notes) = (Vec::new(), Vec::new());

    out(format!("searchlint  n={}  k={k}  mode={}", summary.n, cfg.search.mode));
    let line = |name: &str, v: f64, bv: Option<f64>| -> String {
        let mut s = format!("  {name:10} {v:.3}");
        if let Some(b) = bv { s += &format!("   (baseline {b:.3}, {:+.3})", v - b); }
        s
    };
    out(line("recall", summary.recall, comparable.then(|| base.as_ref().unwrap().summary.recall)));
    out(line("mrr", summary.mrr, comparable.then(|| base.as_ref().unwrap().summary.mrr)));
    for w in golden::warnings(rows).into_iter().take(5) { notes.push(format!("warning: {w}")); }
    let misses: Vec<&Per> = per.iter().filter(|p| !p.hit).collect();
    if !misses.is_empty() {
        let stale = misses.iter().filter(|p| p.why.as_deref() == Some("stale")).count();
        out(format!("  missed {}{}:", misses.len(), if stale > 0 { format!(" ({stale} stale golden: the expected snippet is in no chunk any more; update golden.yaml)") } else { String::new() }));
        for p in misses.iter().take(10) {
            let t = p.top.first().map(|t| format!("{}{}", t.doc, t.page.map(|pg| format!(" p.{pg}")).unwrap_or_default())).unwrap_or("-".into());
            let why = match p.why.as_deref() { Some("stale") => "STALE GOLDEN".to_string(), Some(w) => format!("retrieval: {w}"), None => String::new() };
            out(format!("    - {}   [{why}]  top1: {t}", p.q.chars().take(60).collect::<String>()));
        }
    }
    if let Some(b) = &base {
        if !comparable {
            notes.push("baseline was made with a different config (chunk / embed / search): drop check skipped. Run `searchlint run --baseline` to reset it.".into());
        } else {
            let was: std::collections::HashMap<&str, &Per> = b.per.iter().map(|p| (p.q.as_str(), p)).collect();
            let broken: Vec<&Per> = per.iter().filter(|p| !p.hit && was.get(p.q.as_str()).map(|w| w.hit).unwrap_or(false)).collect();
            let fixed = per.iter().filter(|p| p.hit && was.get(p.q.as_str()).map(|w| !w.hit).unwrap_or(false)).count();
            if fixed > 0 || !broken.is_empty() {
                out(format!("  vs baseline: {fixed} fixed, {} broken{}", broken.len(), if broken.is_empty() { String::new() } else { format!(": {}", broken.iter().take(5).map(|p| p.q.chars().take(40).collect::<String>()).collect::<Vec<_>>().join("; ")) }));
            }
            if summary.recall < b.summary.recall - gate.max_drop {
                problems.push(format!("recall dropped {:.3} (> max_drop {})", b.summary.recall - summary.recall, gate.max_drop));
            }
        }
    }
    if summary.recall < gate.recall { problems.push(format!("recall {:.3} < gate.recall {}", summary.recall, gate.recall)); }
    if summary.mrr < gate.mrr { problems.push(format!("mrr {:.3} < gate.mrr {}", summary.mrr, gate.mrr)); }
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
