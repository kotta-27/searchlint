//! searchlint: lint for retrieval.
//! exit codes: 0 ok, 1 gate failed, 2 cannot evaluate (stale index, bad config, Ollama down)
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
    Draft { #[arg(default_value_t = 20)] n: usize, #[arg(long)] seed: Option<u64> },
    /// mark every draft as ok
    Approve,
    List,
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
    match &ix.meta.model { Some(m) => println!("indexed. embedded {} new chunks with {m}", ix.meta.embedded_new), None => println!("indexed. bm25 only, no embeddings") }
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
                    golden::save(&path, &rows)?; println!("approved {n}");
                }
                GoldenCmd::Draft { n, seed } => {
                    let ix = index::Index::open(&cfg)?;
                    let seed = seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(1));
                    let new = golden::draft(&cfg, &ix.chunks, n, &rows, seed)?;
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
            let mut ix = index::Index::open(&cfg)?;
            if let Some(reason) = ix.staleness()? {
                if reindex { println!("index stale ({reason}): rebuilding"); do_index(&cfg)?; ix = index::Index::open(&cfg)?; }
                else { anyhow::bail!("index is stale: {reason}. Run `searchlint index` (or `searchlint run --index`)"); }
            }
            let mut lines: Vec<String> = Vec::new();
            let (ok, result) = run::run(&cfg, &ix, &rows, &run::Opts { ci, update_baseline: baseline }, &mut |s| if json { lines.push(s) } else { println!("{s}") })?;
            if json {
                let missed: Vec<&str> = result.per.iter().filter(|p| !p.hit).map(|p| p.q.as_str()).collect();
                let mut v = serde_json::to_value(&result)?; v.as_object_mut().unwrap().remove("per"); v["missed"] = serde_json::json!(missed);
                println!("{}", serde_json::to_string_pretty(&v)?);
            }
            Ok(if ok || !ci { 0 } else { 1 })
        }
        Cmd::Serve { .. } => { mcp::serve(cfg)?; Ok(0) }
    }
}
