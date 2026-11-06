//! Embedding providers. `ollama`: HTTP to a local Ollama. `fastembed`: BGE-M3 (ONNX) in-process, no server needed;
//! the model is downloaded once to ~/.cache/searchlint/models. Both return L2-normalized vectors.
use crate::config::Config;
use crate::ollama::Ollama;
use anyhow::{bail, Context, Result};

pub enum Embedder {
    Ollama { client: Ollama, model: String },
    #[cfg(feature = "local-embed")]
    Fast(std::cell::RefCell<fastembed::Bgem3Embedding>),
}

impl Embedder {
    pub fn new(cfg: &Config) -> Result<Embedder> {
        match cfg.embed.provider.as_str() {
            "ollama" => Ok(Embedder::Ollama { client: Ollama::new(&cfg.embed.host), model: cfg.embed.model.clone() }),
            "fastembed" => {
                #[cfg(feature = "local-embed")]
                {
                    use fastembed::{Bgem3Embedding, Bgem3InitOptions, Bgem3Model};
                    let name = cfg.embed.model.to_ascii_lowercase().replace('_', "-");
                    if !matches!(name.as_str(), "bge-m3" | "bgem3" | "bge-m3-q" | "bgem3q") { bail!("fastembed provider supports model bge-m3 (got {})", cfg.embed.model); }
                    let cache = dirs_cache().join("models");
                    std::fs::create_dir_all(&cache)?;
                    let m = Bgem3Embedding::try_new(Bgem3InitOptions::new(Bgem3Model::BGEM3Q).with_cache_dir(cache).with_show_download_progress(true))
                        .context("load BGE-M3 (fastembed)")?;
                    return Ok(Embedder::Fast(std::cell::RefCell::new(m)));
                }
                #[cfg(not(feature = "local-embed"))]
                bail!("this build has no local embeddings (feature local-embed); use provider: ollama")
            }
            other => bail!("embed.provider must be ollama | fastembed (got {other})"),
        }
    }

    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match self {
            Embedder::Ollama { client, model } => client.embed(model, texts),
            #[cfg(feature = "local-embed")]
            Embedder::Fast(m) => {
                let out = m.borrow_mut().embed(texts, Some(16)).map_err(|e| anyhow::anyhow!("fastembed: {e}"))?;
                Ok(out.dense.into_iter().map(normalize).collect())
            }
        }
    }
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-9);
    for a in &mut v { *a /= n; }
    v
}

fn dirs_cache() -> std::path::PathBuf {
    if let Ok(x) = std::env::var("XDG_CACHE_HOME") { return std::path::PathBuf::from(x).join("searchlint"); }
    std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".cache").join("searchlint")
}
