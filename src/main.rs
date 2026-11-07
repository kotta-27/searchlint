//! searchlint: lint for retrieval.
//! exit codes: 0 ok, 1 gate failed, 2 cannot evaluate (stale index, bad config, Ollama down)
mod check;
mod config;
mod embed;
mod golden;
mod index;
mod ingest;
mod mcp;
mod ollama;
mod run;
mod tokenize;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::Config;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "searchlint", version, about = "Lint for retrieval: golden queries, recall/MRR, a CI gate, and an MCP server over the same index.")]
struct Cli {
    /// config file
    #[arg(short, long, default_value = "searchlint.yaml")]
    config: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// write searchlint.yaml
    Init { #[arg(long, default_value = "./docs")] docs: String },
    /// chunk + embed + index (incremental embeddings; bm25 mode needs no embeddings)
    Index,
    /// golden queries: draft N | approve | list
    Golden { #[command(subcommand)] cmd: GoldenCmd },
    /// evaluate against golden.yaml
    Run {
        /// exit 1 when the gate fails
        #[arg(long)] ci: bool,
        /// store this run as the baseline
        #[arg(long)] baseline: bool,
        /// rebuild the index first if it is stale
        #[arg(long)] index: bool,
        /// machine-readable output
        #[arg(long)] json: bool,
    },
    /// MCP server over the same index (stdio)
    Serve { #[arg(long)] mcp: bool },
}

#[derive(Subcommand)]
enum GoldenCmd {
    /// LLM drafts N golden queries (status: draft) -> review golden.yaml
    Draft { #[arg(default_value_t = 20)] n: usize, #[arg(long)] seed: Option<u64>,
        /// only draft from documents that have no golden rows yet
        #[arg(long)] uncovered: bool },
    /// grade the golden set itself: stale or over-generic expects, vague questions, coverage, wording overlap
    Check { /// exit 1 on warnings too (default: only errors)
        #[arg(long)] strict: bool },
    /// mark every draft as ok
    Approve,
    List,
    /// record each row's document version and answer location (after reviewing changes, re-run to accept them)
    Pin,
}

fn main() {
    let cli = Cli::parse();
    std::process::exit(match real_main(cli) {
        Ok(code) => code,
        Err(e) => { eprintln!("searchlint: {e:#}"); 2 }
    });
}

fn progress(i: usize, n: usize) {
    let w = 24; let f = w * i / n.max(1);
    eprint!("\rembed {i}/{n} ▕{}{}▏", "█".repeat(f), " ".repeat(w - f));
    if i >= n { eprintln!(); }
    let _ = std::io::stderr().flush();
}

fn do_index(cfg: &Config) -> Result<i32> {
    let (files, chunks) = ingest::build_chunks(cfg)?;
    if chunks.is_empty() { anyhow::bail!("no chunks: no files matched {:?} under {}", cfg.include, cfg.docs_dir().display()); }
    println!("{} files -> {} chunks", files.len(), chunks.len());
    let ix = index::Index::build(cfg, chunks, &progress)?;
    match &ix.meta.model { Some(m) => println!("indexed in {:.1} s. embedded {} new chunks with {m}", ix.meta.build_sec, ix.meta.embedded_new), None => println!("indexed in {:.1} s. bm25 only, no embeddings", ix.meta.build_sec) }
    Ok(0)
}

fn real_main(cli: Cli) -> Result<i32> {
    if let Cmd::Init { docs } = &cli.cmd {
        if cli.config.exists() { anyhow::bail!("{} exists", cli.config.display()); }
        std::fs::write(&cli.config, Config::default_yaml(docs))?;
        println!("wrote {}  (docs: {docs}). next: searchlint index", cli.config.display());
        return Ok(0);
    }
    let cfg = Config::load(&cli.config)?;
    match cli.cmd {
        Cmd::Init { .. } => unreachable!(),
        Cmd::Index => do_index(&cfg),
        Cmd::Golden { cmd } => {
            let path = cfg.golden_path();
            let mut rows = golden::load(&path)?;
            match cmd {
                GoldenCmd::List => {
                    for r in &rows { println!("[{}] {}  -> {:?}{}", r.status, r.q, r.expect, r.doc.as_ref().map(|d| format!("  ({d})")).unwrap_or_default()); }
                    for w in golden::warnings(&rows) { println!("warning: {w}"); }
                    println!("{} queries ({} draft)", rows.len(), rows.iter().filter(|r| r.status == "draft").count());
                }
                GoldenCmd::Approve => {
                    let n = rows.iter().filter(|r| r.status == "draft").count();
                    for r in &mut rows { r.status = "ok".into(); }
                    let ix = index::Index::open(&cfg)?;
                    let hashes = ingest::doc_hashes(&cfg)?;
                    let mut fresh: Vec<golden::Row> = rows.iter().filter(|r| r.doc_sha.is_none()).cloned().collect();
                    golden::pin(&mut fresh, &ix.chunks, &hashes);
                    for f in fresh { if let Some(r) = rows.iter_mut().find(|r| r.q == f.q && r.doc_sha.is_none()) { *r = f; } }
                    golden::save(&path, &rows)?; println!("approved {n} (pinned to the current document versions)");
                }
                GoldenCmd::Pin => {
                    let ix = index::Index::open(&cfg)?;
                    let hashes = ingest::doc_hashes(&cfg)?;
                    let (pinned, missing) = golden::pin(&mut rows, &ix.chunks, &hashes);
                    golden::save(&path, &rows)?;
                    println!("pinned {pinned} rows to the current documents{}", if missing > 0 { format!("; {missing} rows' expect is in no chunk (left as they were: fix their expect)") } else { String::new() });
                }
                GoldenCmd::Check { strict } => {
                    let ix = index::Index::open(&cfg)?;
                    let rep = check::check(&cfg, &ix, &rows, &mut |s| println!("{s}"))?;
                    return Ok(if rep.errors > 0 || (strict && rep.warnings > 0) { 1 } else { 0 });
                }
                GoldenCmd::Draft { n, seed, uncovered } => {
                    let ix = index::Index::open(&cfg)?;
                    let seed = seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(1));
                    let covered: std::collections::HashSet<String> = rows.iter().filter_map(|r| r.doc.clone()).collect();
                    let sources: Vec<ingest::Chunk> = if uncovered { ix.chunks.iter().filter(|c| !covered.contains(&c.doc)).cloned().collect() } else { ix.chunks.clone() };
                    if sources.is_empty() { println!("every document already has golden rows"); return Ok(0); }
                    let new = golden::draft_from(&cfg, &sources, &ix.chunks, n, &rows, seed)?;
                    let added = new.len();
                    rows.extend(new);
                    golden::save(&path, &rows)?;
                    println!("drafted {added} -> {}. Review them, then: searchlint golden approve", cfg.golden);
                }
            }
            Ok(0)
        }
        Cmd::Run { ci, baseline, index: reindex, json } => {
            let rows: Vec<golden::Row> = golden::load(&cfg.golden_path())?.into_iter().filter(|r| r.status != "draft").collect();
            if rows.is_empty() { anyhow::bail!("no approved golden queries. searchlint golden draft 20 -> review -> searchlint golden approve"); }
            let t_open = std::time::Instant::now();
            let mut ix = index::Index::open(&cfg)?;
            let mut startup_ms = t_open.elapsed().as_secs_f64() * 1000.0;
            if let Some(reason) = ix.staleness()? {
                if reindex { println!("index stale ({reason}): rebuilding"); do_index(&cfg)?; let t = std::time::Instant::now(); ix = index::Index::open(&cfg)?; startup_ms = t.elapsed().as_secs_f64() * 1000.0; }
                else { anyhow::bail!("index is stale: {reason}. Run `searchlint index` (or `searchlint run --index`)"); }
            }
            let mut lines: Vec<String> = Vec::new();
            let (ok, result) = run::run(&cfg, &ix, &rows, &run::Opts { ci, update_baseline: baseline }, &mut |s| if json { lines.push(s) } else { println!("{s}") })?;
            if !json { println!("  startup    {:.0} ms (index open{})", startup_ms, if cfg.needs_embeddings() { " + embedding model load" } else { "" }); }
            if json {
                let missed: Vec<&str> = result.per.iter().filter(|p| !p.hit).map(|p| p.q.as_str()).collect();
                let mut v = serde_json::to_value(&result)?; v.as_object_mut().unwrap().remove("per"); v["missed"] = serde_json::json!(missed);
                v["startup_ms"] = serde_json::json!((startup_ms * 10.0).round() / 10.0); v["index_build_sec"] = serde_json::json!(ix.meta.build_sec);
                println!("{}", serde_json::to_string_pretty(&v)?);
            }
            Ok(if ok || !ci { 0 } else { 1 })
        }
        Cmd::Serve { .. } => { mcp::serve(cfg)?; Ok(0) }
    }
}
