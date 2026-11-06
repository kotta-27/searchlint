//! searchlint.yaml: one file per repo describing the corpus, the index and the gate.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    #[serde(default = "d_size")]
    pub size: usize,
    #[serde(default = "d_overlap")]
    pub overlap: usize,
    #[serde(default = "d_true")]
    pub clean: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    #[serde(default = "d_ollama")]
    pub provider: String,
    pub model: String,
    #[serde(default = "d_host")]
    pub host: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Search {
    #[serde(default = "d_mode")]
    pub mode: String,
    #[serde(default = "d_k")]
    pub k: usize,
    #[serde(default = "d_pool")]
    pub pool: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gate {
    #[serde(default = "d_recall")]
    pub recall: f64,
    #[serde(default)]
    pub mrr: f64,
    #[serde(default = "d_drop")]
    pub max_drop: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "d_docs")]
    pub docs: String,
    #[serde(default = "d_include")]
    pub include: Vec<String>,
    #[serde(default = "d_chunk")]
    pub chunk: Chunk,
    #[serde(default = "d_embed")]
    pub embed: Model,
    #[serde(default = "d_llm")]
    pub llm: Model,
    #[serde(default = "d_search")]
    pub search: Search,
    #[serde(default = "d_gate")]
    pub gate: Gate,
    #[serde(default = "d_golden")]
    pub golden: String,
    #[serde(default = "d_baseline")]
    pub baseline: String,
    #[serde(default = "d_state")]
    pub state: String,
    #[serde(skip)]
    pub root: PathBuf,
}

fn d_size() -> usize { 500 }
fn d_overlap() -> usize { 60 }
fn d_true() -> bool { true }
fn d_ollama() -> String { "ollama".into() }
fn d_host() -> String { "http://localhost:11434".into() }
fn d_mode() -> String { "hybrid".into() }
fn d_k() -> usize { 5 }
fn d_pool() -> usize { 30 }
fn d_recall() -> f64 { 0.9 }
fn d_drop() -> f64 { 0.02 }
fn d_docs() -> String { "./docs".into() }
fn d_include() -> Vec<String> { vec!["**/*.md".into(), "**/*.txt".into(), "**/*.pdf".into()] }
fn d_chunk() -> Chunk { Chunk { size: 500, overlap: 60, clean: true } }
fn d_embed() -> Model { Model { provider: d_ollama(), model: "bge-m3".into(), host: d_host() } }
fn d_llm() -> Model { Model { provider: d_ollama(), model: "qwen3:8b".into(), host: d_host() } }
fn d_search() -> Search { Search { mode: d_mode(), k: 5, pool: 30 } }
fn d_gate() -> Gate { Gate { recall: 0.9, mrr: 0.0, max_drop: 0.02 } }
fn d_golden() -> String { "golden.yaml".into() }
fn d_baseline() -> String { "searchlint.baseline.json".into() }
fn d_state() -> String { ".searchlint".into() }

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let mut c: Config = if path.exists() {
            serde_yaml::from_str(&std::fs::read_to_string(path)?).with_context(|| format!("parse {}", path.display()))?
        } else {
            serde_yaml::from_str("{}")?
        };
        c.root = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        if c.root.as_os_str().is_empty() { c.root = std::env::current_dir()?; }
        c.validate()?;
        Ok(c)
    }
    pub fn default_yaml(docs: &str) -> String {
        format!(
            "# searchlint: retrieval lint config. Commit this file, golden.yaml and searchlint.baseline.json; ignore .searchlint/\n\
docs: {docs}\n\
include: [\"**/*.md\", \"**/*.txt\", \"**/*.pdf\"]\n\
chunk: {{size: 500, overlap: 60, clean: true}}\n\
embed: {{provider: ollama, model: bge-m3, host: http://localhost:11434}}\n\
llm: {{provider: ollama, model: qwen3:8b, host: http://localhost:11434}}\n\
search: {{mode: hybrid, k: 5, pool: 30}}\n\
gate: {{recall: 0.9, mrr: 0.0, max_drop: 0.02}}\n\
golden: golden.yaml\n\
baseline: searchlint.baseline.json\n\
state: .searchlint\n"
        )
    }
    fn validate(&self) -> Result<()> {
        if !(50..=5000).contains(&self.chunk.size) { bail!("chunk.size must be 50..5000"); }
        if self.chunk.overlap * 2 >= self.chunk.size { bail!("chunk.overlap must be < chunk.size / 2"); }
        if !["bm25", "dense", "hybrid"].contains(&self.search.mode.as_str()) { bail!("search.mode must be bm25 | dense | hybrid"); }
        if !(1..=100).contains(&self.search.k) { bail!("search.k must be 1..100"); }
        Ok(())
    }
    pub fn abs(&self, p: &str) -> PathBuf { let pb = PathBuf::from(p); if pb.is_absolute() { pb } else { self.root.join(pb) } }
    pub fn docs_dir(&self) -> PathBuf { self.abs(&self.docs) }
    pub fn state_dir(&self) -> PathBuf { self.abs(&self.state) }
    pub fn golden_path(&self) -> PathBuf { self.abs(&self.golden) }
    pub fn baseline_path(&self) -> PathBuf { self.abs(&self.baseline) }
    pub fn needs_embeddings(&self) -> bool { self.search.mode != "bm25" }
    /// what the index and the baseline depend on: chunking, embedding model, search settings
    pub fn fingerprint(&self) -> String {
        let key = serde_json::json!({
            "chunk": {"size": self.chunk.size, "overlap": self.chunk.overlap, "clean": self.chunk.clean},
            "embed": if self.needs_embeddings() { Some(&self.embed.model) } else { None },
            "search": {"mode": self.search.mode, "k": self.search.k, "pool": self.search.pool},
        });
        hex::encode(Sha256::digest(key.to_string().as_bytes()))[..10].to_string()
    }
}
