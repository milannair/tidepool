use std::collections::HashMap;
use std::io::{Cursor, Read};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use roaring::RoaringBitmap;

use crate::document::Document;
use crate::text::Tokenizer;

const MAGIC: &[u8; 4] = b"TPTI";
const VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct TextIndex {
    pub doc_count: u32,
    pub term_count: u32,
    pub avg_doc_length: f32,
    pub vocab: Vec<String>,
    pub postings: Vec<Vec<(u32, u32)>>,
    pub doc_lengths: Vec<u32>,
    pub vocab_map: HashMap<String, u32>,
}

impl TextIndex {
    pub fn build(docs: &[Document], tokenizer: &dyn Tokenizer) -> Option<Self> {
        let doc_count = docs.len() as u32;
        if doc_count == 0 {
            return None;
        }

        let mut vocab_map: HashMap<String, u32> = HashMap::new();
        let mut vocab: Vec<String> = Vec::new();
        let mut postings: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut doc_lengths: Vec<u32> = vec![0; docs.len()];
        let mut total_len: u64 = 0;

        for (doc_id, doc) in docs.iter().enumerate() {
            let text = doc.text.as_deref().unwrap_or("");
            if text.is_empty() {
                continue;
            }
            let tokens = tokenizer.tokenize(text);
            if tokens.is_empty() {
                continue;
            }

            doc_lengths[doc_id] = tokens.len() as u32;
            total_len += tokens.len() as u64;

            let mut tf_map: HashMap<u32, u32> = HashMap::new();
            for token in tokens {
                let term_id = match vocab_map.get(&token) {
                    Some(&id) => id,
                    None => {
                        let id = vocab.len() as u32;
                        vocab_map.insert(token.clone(), id);
                        vocab.push(token);
                        postings.push(Vec::new());
                        id
                    }
                };
                *tf_map.entry(term_id).or_insert(0) += 1;
            }

            for (term_id, tf) in tf_map {
                if let Some(list) = postings.get_mut(term_id as usize) {
                    list.push((doc_id as u32, tf));
                }
            }
        }

        if vocab.is_empty() {
            return None;
        }

        let avg_doc_length = if doc_count > 0 {
            total_len as f32 / doc_count as f32
        } else {
            0.0
        };

        Some(Self {
            doc_count,
            term_count: vocab.len() as u32,
            avg_doc_length,
            vocab,
            postings,
            doc_lengths,
            vocab_map,
        })
    }

    pub fn search(
        &self,
        tokens: &[String],
        top_k: usize,
        allowed: Option<&RoaringBitmap>,
        k1: f32,
        b: f32,
    ) -> Vec<(u32, f32)> {
        if tokens.is_empty() || self.doc_count == 0 {
            return Vec::new();
        }

        let mut scores: HashMap<u32, f32> = HashMap::new();
        let n = self.doc_count as f32;
        let avgdl = if self.avg_doc_length > 0.0 {
            self.avg_doc_length
        } else {
            1.0
        };

        for token in tokens {
            let Some(&term_id) = self.vocab_map.get(token) else { continue };
            let postings = match self.postings.get(term_id as usize) {
                Some(list) => list,
                None => continue,
            };
            if postings.is_empty() {
                continue;
            }

            let df = postings.len() as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            for &(doc_id, tf) in postings {
                if let Some(allowed) = allowed {
                    if !allowed.contains(doc_id) {
                        continue;
                    }
                }
                let dl = *self.doc_lengths.get(doc_id as usize).unwrap_or(&0) as f32;
                let tf = tf as f32;
                let denom = tf + k1 * (1.0 - b + b * dl / avgdl);
                if denom == 0.0 {
                    continue;
                }
                let score = idf * (tf * (k1 + 1.0)) / denom;
                *scores.entry(doc_id).or_insert(0.0) += score;
            }
        }

        if scores.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<(u32, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if top_k > 0 && results.len() > top_k {
            results.truncate(top_k);
        }
        results
    }

    pub fn marshal_binary(&self) -> Result<Vec<u8>, String> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.write_u32::<LittleEndian>(VERSION).map_err(|e| e.to_string())?;
        buf.write_u32::<LittleEndian>(self.doc_count)
            .map_err(|e| e.to_string())?;
        buf.write_u32::<LittleEndian>(self.term_count)
            .map_err(|e| e.to_string())?;
        buf.write_f32::<LittleEndian>(self.avg_doc_length)
            .map_err(|e| e.to_string())?;

        for term in &self.vocab {
            let bytes = term.as_bytes();
            buf.write_u32::<LittleEndian>(bytes.len() as u32)
                .map_err(|e| e.to_string())?;
            buf.extend_from_slice(bytes);
        }

        for postings in &self.postings {
            buf.write_u32::<LittleEndian>(postings.len() as u32)
                .map_err(|e| e.to_string())?;
            for (doc_id, tf) in postings {
                buf.write_u32::<LittleEndian>(*doc_id)
                    .map_err(|e| e.to_string())?;
                buf.write_u32::<LittleEndian>(*tf)
                    .map_err(|e| e.to_string())?;
            }
        }

        for len in &self.doc_lengths {
            buf.write_u32::<LittleEndian>(*len)
                .map_err(|e| e.to_string())?;
        }

        Ok(buf)
    }

    pub fn load_binary(data: &[u8]) -> Result<Self, String> {
        let mut cursor = Cursor::new(data);
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != MAGIC {
            return Err("invalid text index magic".to_string());
        }

        let version = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        if version != VERSION {
            return Err(format!("unsupported text index version: {}", version));
        }

        let doc_count = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let term_count = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let avg_doc_length = cursor.read_f32::<LittleEndian>().map_err(|e| e.to_string())?;

        let mut vocab = Vec::with_capacity(term_count as usize);
        let mut vocab_map: HashMap<String, u32> = HashMap::new();
        for term_id in 0..term_count {
            let len = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
            let mut buf = vec![0u8; len];
            cursor.read_exact(&mut buf).map_err(|e| e.to_string())?;
            let term = String::from_utf8(buf).map_err(|e| e.to_string())?;
            vocab_map.insert(term.clone(), term_id);
            vocab.push(term);
        }

        let mut postings = Vec::with_capacity(term_count as usize);
        for _ in 0..term_count {
            let list_len = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
            let mut list = Vec::with_capacity(list_len);
            for _ in 0..list_len {
                let doc_id = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                let tf = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                list.push((doc_id, tf));
            }
            postings.push(list);
        }

        let mut doc_lengths = Vec::with_capacity(doc_count as usize);
        for _ in 0..doc_count {
            let len = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
            doc_lengths.push(len);
        }

        Ok(Self {
            doc_count,
            term_count,
            avg_doc_length,
            vocab,
            postings,
            doc_lengths,
            vocab_map,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{DefaultTokenizer, TokenizerConfig};

    #[test]
    fn test_build_and_search() {
        let tokenizer = DefaultTokenizer::new(TokenizerConfig::default());
        let docs = vec![
            Document {
                id: "a".to_string(),
                vector: vec![1.0, 0.0],
                text: Some("hello world".to_string()),
                attributes: None,
            },
            Document {
                id: "b".to_string(),
                vector: vec![0.0, 1.0],
                text: Some("hello rust".to_string()),
                attributes: None,
            },
        ];
        let index = TextIndex::build(&docs, &tokenizer).unwrap();
        let tokens = tokenizer.tokenize("hello");
        let results = index.search(&tokens, 10, None, 1.2, 0.75);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_roundtrip() {
        let tokenizer = DefaultTokenizer::new(TokenizerConfig::default());
        let docs = vec![Document {
            id: "a".to_string(),
            vector: vec![1.0, 0.0],
            text: Some("hello world".to_string()),
            attributes: None,
        }];
        let index = TextIndex::build(&docs, &tokenizer).unwrap();
        let data = index.marshal_binary().unwrap();
        let loaded = TextIndex::load_binary(&data).unwrap();
        assert_eq!(loaded.term_count, index.term_count);
    }
}
