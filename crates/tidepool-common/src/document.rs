use serde::{Deserialize, Serialize};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

use crate::attributes::AttrValue;

pub type Vector = Vec<f32>;

/// Document for JSON API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vector: Vector,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attributes: Option<AttrValue>,
}

/// Document for rkyv serialization (WAL storage).
/// Attributes are stored as JSON bytes to avoid recursive type issues.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct RkyvDocument {
    pub id: String,
    pub vector: Vector,
    pub attributes_json: Vec<u8>,
}

impl From<&Document> for RkyvDocument {
    fn from(doc: &Document) -> Self {
        let attributes_json = match &doc.attributes {
            Some(attrs) => serde_json::to_vec(attrs).unwrap_or_default(),
            None => Vec::new(),
        };
        Self {
            id: doc.id.clone(),
            vector: doc.vector.clone(),
            attributes_json,
        }
    }
}

impl From<RkyvDocument> for Document {
    fn from(doc: RkyvDocument) -> Self {
        let attributes = if doc.attributes_json.is_empty() {
            None
        } else {
            serde_json::from_slice(&doc.attributes_json).ok()
        };
        Self {
            id: doc.id,
            vector: doc.vector,
            attributes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorResult {
    pub id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vector: Vector,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attributes: Option<AttrValue>,
    pub dist: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub vector: Vector,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub ef_search: usize,
    #[serde(default)]
    pub nprobe: usize,
    #[serde(default)]
    pub distance_metric: Option<String>,
    #[serde(default)]
    pub include_vectors: bool,
    #[serde(default)]
    pub filters: Option<AttrValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub results: Vec<VectorResult>,
    pub namespace: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertRequest {
    pub vectors: Vec<Document>,
    #[serde(default)]
    pub distance_metric: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertResponse {
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRequest {
    pub ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResponse {
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceInfo {
    pub namespace: String,
    pub approx_count: i64,
    pub dimensions: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_compaction: Option<bool>,
}
