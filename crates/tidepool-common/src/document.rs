use serde::{Deserialize, Serialize};

pub type Vector = Vec<f32>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vector: Vector,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attributes: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorResult {
    pub id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vector: Vector,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attributes: Option<serde_json::Value>,
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
    pub distance_metric: Option<String>,
    #[serde(default)]
    pub include_vectors: bool,
    #[serde(default)]
    pub filters: Option<serde_json::Value>,
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
}
