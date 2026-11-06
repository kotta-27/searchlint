//! Japanese word segmentation with lindera (IPADIC, embedded in the binary). Output is space-joined surface forms,
//! which tantivy's default tokenizer then splits: the same "pre-tokenize, then whitespace index" trick as the Python version.
use anyhow::Result;
use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use std::borrow::Cow;
use std::sync::LazyLock;

static SEG: LazyLock<Segmenter> = LazyLock::new(|| {
    let dict = load_dictionary("embedded://ipadic").expect("embedded ipadic dictionary");
    Segmenter::new(Mode::Normal, dict, None)
});

pub fn tokens(text: &str) -> Result<Vec<String>> {
    let toks = SEG.segment(Cow::Borrowed(text)).map_err(|e| anyhow::anyhow!("lindera: {e}"))?;
    Ok(toks.into_iter().map(|t| t.surface.to_string()).filter(|s| !s.trim().is_empty()).collect())
}

pub fn joined(text: &str) -> Result<String> { Ok(tokens(text)?.join(" ")) }
