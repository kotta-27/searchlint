//! Ollama client: embeddings and chat (chat is only used to draft golden queries).
use anyhow::{Context, Result};
use serde_json::{json, Value};

pub struct Ollama { host: String, c: reqwest::blocking::Client }

impl Ollama {
    pub fn new(host: &str) -> Ollama {
        Ollama { host: host.trim_end_matches('/').to_string(), c: reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(600)).build().unwrap() }
    }
    /// L2-normalized vectors
    pub fn embed(&self, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let r: Value = self.c.post(format!("{}/api/embed", self.host)).json(&json!({"model": model, "input": texts})).send()
            .with_context(|| format!("cannot reach Ollama at {}", self.host))?.error_for_status()?.json()?;
        let mut out = Vec::new();
        for v in r["embeddings"].as_array().context("no embeddings in response")? {
            let mut x: Vec<f32> = v.as_array().unwrap().iter().map(|f| f.as_f64().unwrap() as f32).collect();
            let n = x.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-9);
            for a in &mut x { *a /= n; }
            out.push(x);
        }
        Ok(out)
    }
    pub fn chat_json(&self, model: &str, prompt: &str, max_tokens: u32) -> Result<Option<Value>> {
        let prompt = if model.contains("qwen3") { format!("{prompt}\n/no_think") } else { prompt.to_string() };
        let body = json!({"model": model, "messages": [{"role": "user", "content": prompt}], "stream": false, "think": false, "format": "json",
            "keep_alive": "30m", "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": max_tokens}});
        let r: Value = self.c.post(format!("{}/api/chat", self.host)).json(&body).send()
            .with_context(|| format!("cannot reach Ollama at {}", self.host))?.error_for_status()?.json()?;
        let mut content = r["message"]["content"].as_str().unwrap_or("").to_string();
        if let Some(i) = content.rfind("</think>") { content = content[i + 8..].to_string(); }
        Ok(serde_json::from_str(content.trim()).ok())
    }
}
