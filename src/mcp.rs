//! MCP server over stdio, implemented directly on JSON-RPC (no SDK dependency): initialize, tools/list, tools/call.
//! Tools: search / quality / list_documents / recent_queries / reload. Results carry structuredContent.
use crate::config::Config;
use crate::index::Index;
use anyhow::Result;
use serde_json::{json, Value};
use std::io::{BufRead, Write};

pub fn serve(cfg: Config) -> Result<()> {
    let mut index = Index::open(&cfg)?;
    let log_path = cfg.state_dir().join("queries.jsonl");
    let mode = cfg.search.mode.clone();
    let hint = match mode.as_str() {
        "bm25" => "Keyword search: use the exact words the documents would use, not paraphrases.",
        "dense" => "Semantic search: natural-language questions work; exact identifiers may be missed.",
        _ => "Keyword + semantic search fused by rank.",
    };
    let tools = json!([
        {"name": "search", "description": format!("Search the project's documents ({mode}). {hint} Returns ranked chunks with document, page and chunk id, plus `trust`: \
how far to rely on these results. If trust.ok is false, say why (trust.reason) before answering from the hits; an empty or weak result \
from a well-measured index (trust.ok, high recall) more likely means the documents do not cover the question."),
         "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 500}, "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": cfg.search.k},
                         "doc": {"type": "string", "description": "restrict to one document path as returned by list_documents"}}, "required": ["query"]}},
        {"name": "quality", "description": "Whether to trust search: ok (fresh index + measured recall above the gate), recall@k, MRR, n golden queries, when measured, and the reason if not ok.",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "list_documents", "description": "Documents in the index with chunk and page counts. Use it to tell 'no such document' from 'wrong words'.",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "recent_queries", "description": "Recent queries made through this server with hit counts: candidates for new golden queries.",
         "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50}}}},
        {"name": "reload", "description": "Reload the index from disk after `searchlint index` ran.", "inputSchema": {"type": "object", "properties": {}}},
    ]);

    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let req: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = req.get("id").cloned();
        let method = req["method"].as_str().unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(json!({}));
        let result: Option<Value> = match method {
            "initialize" => Some(json!({"protocolVersion": params["protocolVersion"].as_str().unwrap_or("2025-06-18"), "capabilities": {"tools": {}},
                "serverInfo": {"name": "searchlint", "version": env!("CARGO_PKG_VERSION")},
                "instructions": "Search the project's documents. Call quality first: it says whether the index is fresh and how often golden queries are found."})),
            "notifications/initialized" | "notifications/cancelled" => None,
            "ping" => Some(json!({})),
            "tools/list" => Some(json!({"tools": tools})),
            "tools/call" => {
                let name = params["name"].as_str().unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                match call(&cfg, &mut index, &log_path, name, &args) {
                    Ok(v) => Some(json!({"content": [{"type": "text", "text": v.to_string()}], "structuredContent": v, "isError": false})),
                    Err(e) => Some(json!({"content": [{"type": "text", "text": format!("{e:#}")}], "isError": true})),
                }
            }
            _ => { if id.is_some() { Some(json!({"error": {"code": -32601, "message": format!("unknown method {method}")}})) } else { None } }
        };
        if let (Some(id), Some(result)) = (id, result) {
            let msg = if let Some(err) = result.get("error") { json!({"jsonrpc": "2.0", "id": id, "error": err}) } else { json!({"jsonrpc": "2.0", "id": id, "result": result}) };
            writeln!(out, "{}", msg)?; out.flush()?;
        }
    }
    Ok(())
}

fn quality(cfg: &Config, index: &Index) -> Result<Value> {
    let stale = index.staleness()?;
    let last = cfg.state_dir().join("last.json");
    if !last.exists() {
        return Ok(json!({"ok": false, "measured": false, "stale": stale.is_some(), "reason": format!("no searchlint run yet{}", stale.map(|s| format!("; {s}")).unwrap_or_default())}));
    }
    let r: Value = serde_json::from_str(&std::fs::read_to_string(&last)?)?;
    let s = &r["summary"];
    let mut reasons: Vec<String> = Vec::new();
    if let Some(st) = &stale { reasons.push(st.clone()); }
    if r["fingerprint"].as_str() != Some(&cfg.fingerprint()) { reasons.push("last run used a different config".into()); }
    let recall = s["recall"].as_f64().unwrap_or(0.0);
    if recall < cfg.gate.recall { reasons.push(format!("recall {recall:.3} below gate {}", cfg.gate.recall)); }
    Ok(json!({"ok": reasons.is_empty(), "measured": true, "stale": stale.is_some(), "at": r["at"], "k": r["k"], "mode": r["mode"], "n": s["n"],
              "recall": (recall * 1000.0).round() / 1000.0, "p50_ms": s["p50_ms"], "p95_ms": s["p95_ms"], "mrr": (s["mrr"].as_f64().unwrap_or(0.0) * 1000.0).round() / 1000.0,
              "gate_recall": cfg.gate.recall, "reason": if reasons.is_empty() { Value::Null } else { json!(reasons.join("; ")) }}))
}

fn call(cfg: &Config, index: &mut Index, log_path: &std::path::Path, name: &str, args: &Value) -> Result<Value> {
    match name {
        "search" => {
            let q = args["query"].as_str().unwrap_or("").trim().to_string();
            if q.is_empty() { return Ok(json!({"query": "", "mode": cfg.search.mode, "hits": [], "note": "empty query"})); }
            let k = args["k"].as_u64().map(|k| k.clamp(1, 20) as usize).unwrap_or(cfg.search.k);
            let doc = args["doc"].as_str().map(|s| s.to_string());
            let hits = index.search(&q, Some(k), doc.as_deref())?;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log_path) {
                let _ = writeln!(f, "{}", json!({"at": chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(), "q": q, "n_hits": hits.len(), "top": hits.iter().take(3).map(|h| &h.id).collect::<Vec<_>>()}));
            }
            let qd = quality(cfg, index)?;
            let trust = json!({"ok": qd["ok"], "stale": qd["stale"], "recall": qd.get("recall").cloned().unwrap_or(Value::Null),
                               "n_golden": qd.get("n").cloned().unwrap_or(Value::Null), "measured_at": qd.get("at").cloned().unwrap_or(Value::Null), "reason": qd["reason"]});
            Ok(json!({"query": q, "mode": cfg.search.mode, "k": k, "doc": doc, "trust": trust,
                "hits": hits.iter().enumerate().map(|(i, h)| json!({"rank": i + 1, "id": h.id, "doc": h.doc, "page": h.page, "text": h.text})).collect::<Vec<_>>()}))
        }
        "quality" => quality(cfg, index),
        "list_documents" => Ok(json!({"built": index.meta.built, "documents": index.documents()})),
        "recent_queries" => {
            let limit = args["limit"].as_u64().unwrap_or(50) as usize;
            let mut seen: Vec<(String, Value)> = Vec::new();
            if let Ok(text) = std::fs::read_to_string(log_path) {
                for l in text.lines() {
                    if let Ok(r) = serde_json::from_str::<Value>(l) {
                        let q = r["q"].as_str().unwrap_or("").to_string();
                        let count = seen.iter().find(|(k, _)| *k == q).map(|(_, v)| v["count"].as_u64().unwrap_or(0)).unwrap_or(0) + 1;
                        seen.retain(|(k, _)| *k != q);
                        seen.push((q.clone(), json!({"q": q, "last": r["at"], "n_hits": r["n_hits"], "count": count})));
                    }
                }
            }
            let n = seen.len();
            Ok(json!({"queries": seen.into_iter().skip(n.saturating_sub(limit)).map(|(_, v)| v).collect::<Vec<_>>()}))
        }
        "reload" => { *index = Index::open(cfg)?; Ok(json!({"built": index.meta.built, "chunks": index.meta.chunks})) }
        _ => anyhow::bail!("unknown tool {name}"),
    }
}
