//! Index: tantivy (BM25 over lindera tokens) and, unless search.mode is bm25, a flat f32 matrix for exact
//! inner-product search. Embeddings are cached per (model, chunk text) under the state dir.
//! index.json records the index fingerprint (chunking + chunker version + embedding model) and the documents manifest so `run` can tell when the index is stale.
use crate::config::Config;
use crate::ingest::{docs_manifest, Chunk};
use crate::embed::Embedder;
use crate::tokenize;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, STORED, TEXT};
use tantivy::{doc, Index as TIndex, ReloadPolicy};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Meta {
    pub built: String,
    pub chunks: usize,
    pub files: usize,
    pub embedded_new: usize,
    pub model: Option<String>,
    pub fingerprint: String,
    pub docs: String,
    pub dims: usize,
}

pub struct Index {
    pub cfg: Config,
    pub chunks: Vec<Chunk>,
    pub meta: Meta,
    tindex: TIndex,
    f_toks: Field,
    f_row: Field,
    dense: Option<Vec<f32>>, // row-major chunks x dims
    embedder: Option<Embedder>,
}

fn paths(cfg: &Config) -> (PathBuf, PathBuf, PathBuf, PathBuf, PathBuf) {
    let s = cfg.state_dir();
    (s.join("chunks.json"), s.join("tantivy"), s.join("dense.f32"), s.join("embed-cache"), s.join("index.json"))
}

fn schema() -> (Schema, Field, Field) {
    let mut b = Schema::builder();
    let f_toks = b.add_text_field("toks", TEXT);
    let f_row = b.add_u64_field("row", STORED);
    (b.build(), f_toks, f_row)
}

impl Index {
    pub fn open(cfg: &Config) -> Result<Index> {
        let (chunks_p, tant_p, dense_p, _, meta_p) = paths(cfg);
        if !meta_p.exists() { bail!("no index yet: run `searchlint index`"); }
        let meta: Meta = serde_json::from_str(&fs::read_to_string(&meta_p)?)?;
        let chunks: Vec<Chunk> = serde_json::from_str(&fs::read_to_string(&chunks_p)?)?;
        let (sch, f_toks, f_row) = schema();
        let tindex = TIndex::open_in_dir(&tant_p).context("open tantivy index")?;
        let _ = sch;
        let mut dense = None;
        if cfg.needs_embeddings() && dense_p.exists() {
            let bytes = fs::read(&dense_p)?;
            dense = Some(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec());
        }
        let embedder = if cfg.needs_embeddings() { Some(Embedder::new(cfg)?) } else { None };
        Ok(Index { cfg: cfg.clone(), chunks, meta, tindex, f_toks, f_row, dense, embedder })
    }

    /// None when fresh, else a reason
    pub fn staleness(&self) -> Result<Option<String>> {
        if self.cfg.needs_embeddings() && self.dense.is_none() { return Ok(Some("index was built without embeddings (bm25)".into())); }
        if self.meta.fingerprint != self.cfg.index_fingerprint() { return Ok(Some("chunking or embedding config (or searchlint version) changed since the index was built".into())); }
        if self.meta.docs != docs_manifest(&self.cfg)?.0 { return Ok(Some("documents changed since the index was built".into())); }
        Ok(None)
    }

    pub fn build(cfg: &Config, chunks: Vec<Chunk>, progress: &dyn Fn(usize, usize)) -> Result<Index> {
        let (chunks_p, tant_p, dense_p, cache_p, meta_p) = paths(cfg);
        fs::create_dir_all(cfg.state_dir())?;
        fs::write(&chunks_p, serde_json::to_string(&chunks)?)?;
        // tantivy
        if tant_p.exists() { fs::remove_dir_all(&tant_p)?; }
        fs::create_dir_all(&tant_p)?;
        let (sch, f_toks, f_row) = schema();
        let tindex = TIndex::create_in_dir(&tant_p, sch)?;
        let mut w = tindex.writer(64_000_000)?;
        for (i, c) in chunks.iter().enumerate() {
            w.add_document(doc!(f_toks => tokenize::joined(&c.text)?, f_row => i as u64))?;
        }
        w.commit()?;
        // dense
        let mut embedded_new = 0usize;
        let mut dims = 0usize;
        let mut dense = None;
        if cfg.needs_embeddings() {
            let model = format!("{}:{}", cfg.embed.provider, cfg.embed.model);
            let embedder = Embedder::new(cfg)?;
            let mut cache = Cache::load(&cache_p, &model)?;
            let need: Vec<&Chunk> = chunks.iter().filter(|c| !cache.has(&c.sha)).collect();
            for (i, batch) in need.chunks(64).enumerate() {
                let vecs = embedder.embed(&batch.iter().map(|c| c.text.clone()).collect::<Vec<_>>())?;
                for (c, v) in batch.iter().zip(vecs) { cache.put(&c.sha, v); }
                progress(((i + 1) * 64).min(need.len()), need.len());
            }
            embedded_new = need.len();
            cache.save(&cache_p)?;
            dims = cache.dims;
            let mut mat: Vec<f32> = Vec::with_capacity(chunks.len() * dims);
            for c in &chunks { mat.extend_from_slice(cache.get(&c.sha).unwrap()); }
            fs::write(&dense_p, bytemuck::cast_slice::<f32, u8>(&mat))?;
            dense = Some(mat);
        } else if dense_p.exists() {
            fs::remove_file(&dense_p)?;
        }
        let (docs_hash, files) = docs_manifest(cfg)?;
        let meta = Meta {
            built: chrono::Local::now().format("%Y-%m-%dT%H:%M").to_string(), chunks: chunks.len(), files: files.len(), embedded_new,
            model: cfg.needs_embeddings().then(|| cfg.embed.model.clone()), fingerprint: cfg.index_fingerprint(), docs: docs_hash, dims,
        };
        fs::write(&meta_p, serde_json::to_string_pretty(&meta)?)?;
        let embedder = if cfg.needs_embeddings() { Some(Embedder::new(cfg)?) } else { None };
        Ok(Index { cfg: cfg.clone(), chunks, meta, tindex, f_toks, f_row, dense, embedder })
    }

    pub fn bm25(&self, q: &str, n: usize) -> Result<Vec<usize>> {
        let toks = tokenize::tokens(q)?;
        if toks.is_empty() { return Ok(vec![]); }
        let reader = self.tindex.reader_builder().reload_policy(ReloadPolicy::Manual).try_into()?;
        let searcher = reader.searcher();
        let parser = QueryParser::for_index(&self.tindex, vec![self.f_toks]);
        let qs = toks.iter().map(|t| format!("\"{}\"", t.replace('"', ""))).collect::<Vec<_>>().join(" ");
        let query = parser.parse_query_lenient(&qs).0;
        let top = searcher.search(&query, &TopDocs::with_limit(n))?;
        let mut out = Vec::new();
        for (_, addr) in top {
            let d: tantivy::TantivyDocument = searcher.doc(addr)?;
            if let Some(v) = d.get_first(self.f_row).and_then(|v| tantivy::schema::Value::as_u64(&v)) { out.push(v as usize); }
        }
        Ok(out)
    }

    pub fn dense_rank(&self, q: &str, n: usize) -> Result<Vec<usize>> {
        let (Some(mat), Some(embedder)) = (&self.dense, &self.embedder) else { bail!("index was built without embeddings: run `searchlint index`") };
        let qv = embedder.embed(&[q.to_string()])?.remove(0);
        let d = self.meta.dims;
        let mut scores: Vec<(f32, usize)> = (0..self.chunks.len()).map(|i| (mat[i * d..(i + 1) * d].iter().zip(&qv).map(|(a, b)| a * b).sum::<f32>(), i)).collect();
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        Ok(scores.into_iter().take(n).map(|(_, i)| i).collect())
    }

    pub fn search(&self, q: &str, k: Option<usize>, doc: Option<&str>) -> Result<Vec<&Chunk>> {
        let s = &self.cfg.search;
        let k = k.unwrap_or(s.k);
        if q.trim().is_empty() { return Ok(vec![]); }
        let n = if doc.is_some() { s.pool } else { k };
        let idx: Vec<usize> = match s.mode.as_str() {
            "bm25" => self.bm25(q, n)?,
            "dense" => self.dense_rank(q, n)?,
            _ => {
                let mut score: HashMap<usize, f64> = HashMap::new();
                for lst in [self.bm25(q, s.pool)?, self.dense_rank(q, s.pool)?] {
                    for (r, i) in lst.into_iter().enumerate() { *score.entry(i).or_default() += 1.0 / (60.0 + r as f64); }
                }
                let mut v: Vec<(usize, f64)> = score.into_iter().collect();
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
                v.into_iter().map(|(i, _)| i).collect()
            }
        };
        let mut hits: Vec<&Chunk> = idx.into_iter().map(|i| &self.chunks[i]).collect();
        if let Some(d) = doc { hits.retain(|c| c.doc == d); }
        hits.truncate(k);
        Ok(hits)
    }

    pub fn documents(&self) -> Vec<serde_json::Value> {
        let mut m: HashMap<&str, (usize, u32)> = HashMap::new();
        for c in &self.chunks { let e = m.entry(&c.doc).or_default(); e.0 += 1; e.1 = e.1.max(c.page.unwrap_or(0)); }
        let mut v: Vec<_> = m.into_iter().map(|(d, (n, p))| serde_json::json!({"doc": d, "chunks": n, "pages": p})).collect();
        v.sort_by_key(|x| x["doc"].as_str().unwrap_or("").to_string());
        v
    }
}

/// embedding cache: keys file (one "model:sha" per line) + flat f32 file in the same order
struct Cache { model: String, dims: usize, map: HashMap<String, Vec<f32>> }
impl Cache {
    fn load(dir: &PathBuf, model: &str) -> Result<Cache> {
        let mut c = Cache { model: model.to_string(), dims: 0, map: HashMap::new() };
        let (kp, vp) = (dir.join("keys.txt"), dir.join("vecs.f32"));
        if kp.exists() && vp.exists() {
            let keys: Vec<String> = fs::read_to_string(&kp)?.lines().map(|s| s.to_string()).collect();
            let bytes = fs::read(&vp)?;
            let vecs: &[f32] = bytemuck::cast_slice(&bytes);
            if !keys.is_empty() && vecs.len() % keys.len() == 0 {
                let d = vecs.len() / keys.len();
                for (i, k) in keys.iter().enumerate() {
                    if let Some(sha) = k.strip_prefix(&format!("{model}:")) { c.map.insert(sha.to_string(), vecs[i * d..(i + 1) * d].to_vec()); c.dims = d; }
                }
            }
        }
        Ok(c)
    }
    fn has(&self, sha: &str) -> bool { self.map.contains_key(sha) }
    fn get(&self, sha: &str) -> Option<&Vec<f32>> { self.map.get(sha) }
    fn put(&mut self, sha: &str, v: Vec<f32>) { self.dims = v.len(); self.map.insert(sha.to_string(), v); }
    fn save(&self, dir: &PathBuf) -> Result<()> {
        fs::create_dir_all(dir)?;
        let mut keys = String::new();
        let mut flat: Vec<f32> = Vec::with_capacity(self.map.len() * self.dims);
        for (sha, v) in &self.map { keys.push_str(&format!("{}:{}\n", self.model, sha)); flat.extend_from_slice(v); }
        fs::write(dir.join("keys.txt"), keys)?;
        fs::write(dir.join("vecs.f32"), bytemuck::cast_slice::<f32, u8>(&flat))?;
        Ok(())
    }
}
