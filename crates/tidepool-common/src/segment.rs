use std::fs::File;
use std::io::Read;
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use hex::encode as hex_encode;
use memmap2::Mmap;
use roaring::RoaringBitmap;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::task;
use tokio::fs;

use crate::attributes::AttrValue;
use crate::document::Document;
use crate::index::hnsw::{HnswIndex, ResultItem, DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH, DEFAULT_M};
use crate::storage::{segment_index_path, segment_path, Store, StorageError};
use crate::vector::{distance, DistanceMetric};

#[derive(Debug, Clone)]
pub struct SegmentData {
    pub ids: Vec<String>,
    pub vectors: Vec<Vec<f32>>,
    pub attributes: Vec<Option<AttrValue>>,
    pub dimensions: usize,
    pub index: Option<HnswIndex>,
    pub filters: Option<FilterIndex>,
}

/// Internal struct for storing segment attributes.
/// Attributes are stored as JSON bytes to avoid recursive type issues with rkyv.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
struct SegmentAttr {
    id: String,
    /// JSON-serialized attributes (or empty vec if None)
    attributes_json: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct WriterOptions {
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub metric: DistanceMetric,
}

impl Default for WriterOptions {
    fn default() -> Self {
        Self {
            hnsw_m: DEFAULT_M,
            hnsw_ef_construction: DEFAULT_EF_CONSTRUCTION,
            hnsw_ef_search: DEFAULT_EF_SEARCH,
            metric: DistanceMetric::Cosine,
        }
    }
}

#[derive(Clone)]
pub struct Writer<S: Store> {
    storage: S,
    namespace: String,
    opts: WriterOptions,
}

impl<S: Store> Writer<S> {
    pub fn new(storage: S, namespace: impl Into<String>) -> Self {
        Self::new_with_options(storage, namespace, WriterOptions::default())
    }

    pub fn new_with_options(storage: S, namespace: impl Into<String>, opts: WriterOptions) -> Self {
        Self {
            storage,
            namespace: namespace.into(),
            opts,
        }
    }

    pub async fn write_segment(&self, docs: &[Document]) -> Result<Option<ManifestSegment>, StorageError> {
        if docs.is_empty() {
            return Ok(None);
        }

        let mut dimensions = 0usize;
        for doc in docs {
            if !doc.vector.is_empty() {
                dimensions = doc.vector.len();
                break;
            }
        }
        if dimensions == 0 {
            return Err(StorageError::Other("no vectors found in documents".to_string()));
        }
        for (i, doc) in docs.iter().enumerate() {
            if doc.vector.is_empty() {
                return Err(StorageError::Other(format!("document {} has empty vector", i)));
            }
            if doc.vector.len() != dimensions {
                return Err(StorageError::Other(format!(
                    "document {} vector dimension mismatch: got {} want {}",
                    i,
                    doc.vector.len(),
                    dimensions
                )));
            }
        }

        let segment_id = uuid::Uuid::new_v4().to_string();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TPVS");
        buf.write_u32::<LittleEndian>(1)
            .map_err(|e| StorageError::Other(format!("write version: {}", e)))?;
        buf.write_u32::<LittleEndian>(docs.len() as u32)
            .map_err(|e| StorageError::Other(format!("write vector count: {}", e)))?;
        buf.write_u32::<LittleEndian>(dimensions as u32)
            .map_err(|e| StorageError::Other(format!("write dimensions: {}", e)))?;

        let mut attr_data = Vec::with_capacity(docs.len());

        for doc in docs {
            for v in &doc.vector {
                buf.write_f32::<LittleEndian>(*v)
                    .map_err(|e| StorageError::Other(format!("write vector: {}", e)))?;
            }
            let attributes_json = match &doc.attributes {
                Some(attrs) => serde_json::to_vec(attrs)
                    .map_err(|e| StorageError::Other(format!("serialize attributes: {}", e)))?,
                None => Vec::new(),
            };
            attr_data.push(SegmentAttr {
                id: doc.id.clone(),
                attributes_json,
            });
        }

        let attr_bytes = rkyv::to_bytes::<_, 256>(&attr_data)
            .map_err(|e| StorageError::Other(format!("serialize attributes: {}", e)))?;
        let attr_bytes = attr_bytes.as_ref();
        buf.write_u32::<LittleEndian>(attr_bytes.len() as u32)
            .map_err(|e| StorageError::Other(format!("write attr length: {}", e)))?;
        buf.extend_from_slice(&attr_bytes);

        let mut hnsw = HnswIndex::new(
            self.opts.hnsw_m,
            self.opts.hnsw_ef_construction,
            self.opts.hnsw_ef_search,
            self.opts.metric,
        );
        for (i, doc) in docs.iter().enumerate() {
            hnsw.insert(i, doc.vector.clone());
        }
        let hnsw_data = hnsw
            .marshal_binary()
            .map_err(|e| StorageError::Other(format!("serialize HNSW: {}", e)))?;

        let segment_key = segment_path(&self.namespace, &segment_id);
        self.storage.put(&segment_key, buf).await?;

        let index_key = segment_index_path(&self.namespace, &segment_id);
        if let Err(err) = self.storage.put(&index_key, hnsw_data).await {
            let _ = self.storage.delete(&segment_key).await;
            return Err(err);
        }

        Ok(Some(ManifestSegment {
            id: segment_id,
            segment_key,
            doc_count: docs.len() as i64,
            dimensions,
        }))
    }
}

#[derive(Debug, Clone)]
pub struct ReaderOptions {
    pub hnsw_ef_search: usize,
}

impl Default for ReaderOptions {
    fn default() -> Self {
        Self {
            hnsw_ef_search: DEFAULT_EF_SEARCH,
        }
    }
}

#[derive(Clone)]
pub struct Reader<S: Store> {
    storage: S,
    namespace: String,
    cache_dir: Option<String>,
    hnsw_ef_search: usize,
}

impl<S: Store> Reader<S> {
    pub fn new(storage: S, namespace: impl Into<String>, cache_dir: Option<String>) -> Self {
        Self::new_with_options(storage, namespace, cache_dir, ReaderOptions::default())
    }

    pub fn new_with_options(
        storage: S,
        namespace: impl Into<String>,
        cache_dir: Option<String>,
        opts: ReaderOptions,
    ) -> Self {
        if let Some(dir) = &cache_dir {
            let _ = std::fs::create_dir_all(dir);
        }
        Self {
            storage,
            namespace: namespace.into(),
            cache_dir,
            hnsw_ef_search: opts.hnsw_ef_search,
        }
    }

    pub async fn read_segment(&self, segment_key: &str) -> Result<SegmentData, StorageError> {
        let data = self.get_segment_data(segment_key).await?;
        let mut cursor = std::io::Cursor::new(data.as_slice());
        let mut magic = [0u8; 4];
        cursor
            .read_exact(&mut magic)
            .map_err(|e| StorageError::Other(format!("read magic: {}", e)))?;
        if &magic != b"TPVS" {
            return Err(StorageError::Other("invalid segment file format".to_string()));
        }
        let version = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| StorageError::Other(format!("read version: {}", e)))?;
        if version != 1 {
            return Err(StorageError::Other(format!("unsupported segment version: {}", version)));
        }
        let num_vectors = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| StorageError::Other(format!("read vector count: {}", e)))? as usize;
        let dimensions = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| StorageError::Other(format!("read dimensions: {}", e)))? as usize;
        if dimensions == 0 {
            return Err(StorageError::Other("invalid segment dimensions".to_string()));
        }

        let mut vectors = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let mut vec = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                let v = cursor
                    .read_f32::<LittleEndian>()
                    .map_err(|e| StorageError::Other(format!("read vector: {}", e)))?;
                vec.push(v);
            }
            vectors.push(vec);
        }

        let attr_len = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| StorageError::Other(format!("read attr length: {}", e)))? as usize;
        if attr_len == 0 {
            return Err(StorageError::Other("invalid attribute length".to_string()));
        }
        let mut attr_bytes = vec![0u8; attr_len];
        cursor
            .read_exact(&mut attr_bytes)
            .map_err(|e| StorageError::Other(format!("read attributes: {}", e)))?;
        // SAFETY: We trust our own serialized data format
        let archived = unsafe { rkyv::archived_root::<Vec<SegmentAttr>>(&attr_bytes) };
        let attr_data: Vec<SegmentAttr> = archived
            .deserialize(&mut rkyv::Infallible)
            .map_err(|e| StorageError::Other(format!("deserialize attributes: {}", e)))?;
        if attr_data.len() != num_vectors {
            return Err(StorageError::Other(format!(
                "attribute count mismatch: got {} want {}",
                attr_data.len(),
                num_vectors
            )));
        }

        let mut ids = Vec::with_capacity(num_vectors);
        let mut attrs = Vec::with_capacity(num_vectors);
        for attr in attr_data {
            ids.push(attr.id);
            let attr_value = if attr.attributes_json.is_empty() {
                None
            } else {
                Some(serde_json::from_slice(&attr.attributes_json)
                    .map_err(|e| StorageError::Other(format!("deserialize attributes: {}", e)))?)
            };
            attrs.push(attr_value);
        }

        let mut seg = SegmentData {
            ids,
            vectors,
            attributes: attrs,
            dimensions,
            index: None,
            filters: None,
        };

        if let Some(segment_id) = segment_id_from_key(segment_key) {
            let index_key = segment_index_path(&self.namespace, &segment_id);
            if self.storage.exists(&index_key).await.unwrap_or(false) {
                if let Ok(index_data) = self.get_index_data(&index_key).await {
                    if let Ok(hnsw) =
                        HnswIndex::load_binary(index_data.as_slice(), &seg.vectors, self.hnsw_ef_search)
                    {
                        seg.index = Some(hnsw);
                    }
                }
            }
        }

        seg.filters = build_filter_index(&seg.attributes);
        Ok(seg)
    }

    async fn get_segment_data(&self, segment_key: &str) -> Result<SegmentBytes, StorageError> {
        if let Some(dir) = &self.cache_dir {
            let cache_path = format!("{}/{}", dir, cache_key(segment_key, "tpvs"));
            if let Some(mmap) = map_cached_file(cache_path.clone()).await {
                return Ok(SegmentBytes::Mapped(mmap));
            }
            if let Ok(data) = fs::read(&cache_path).await {
                return Ok(SegmentBytes::Owned(data));
            }
        }
        let data = self.storage.get(segment_key).await?;
        if let Some(dir) = &self.cache_dir {
            let cache_path = format!("{}/{}", dir, cache_key(segment_key, "tpvs"));
            let _ = fs::write(cache_path, &data).await;
        }
        Ok(SegmentBytes::Owned(data))
    }

    async fn get_index_data(&self, index_key: &str) -> Result<SegmentBytes, StorageError> {
        if let Some(dir) = &self.cache_dir {
            let cache_path = format!("{}/{}", dir, cache_key(index_key, "hnsw"));
            if let Some(mmap) = map_cached_file(cache_path.clone()).await {
                return Ok(SegmentBytes::Mapped(mmap));
            }
            if let Ok(data) = fs::read(&cache_path).await {
                return Ok(SegmentBytes::Owned(data));
            }
        }
        let data = self.storage.get(index_key).await?;
        if let Some(dir) = &self.cache_dir {
            let cache_path = format!("{}/{}", dir, cache_key(index_key, "hnsw"));
            let _ = fs::write(cache_path, &data).await;
        }
        Ok(SegmentBytes::Owned(data))
    }
}

enum SegmentBytes {
    Owned(Vec<u8>),
    Mapped(Mmap),
}

impl SegmentBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            SegmentBytes::Owned(buf) => buf.as_slice(),
            SegmentBytes::Mapped(mmap) => mmap,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScoredResult {
    pub index: usize,
    pub dist: f32,
}

impl SegmentData {
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
        filters: Option<&AttrValue>,
        ef_search: usize,
    ) -> Vec<ScoredResult> {
        let top_k = if top_k == 0 { self.vectors.len() } else { top_k };

        if filters.is_none() {
            if let Some(index) = &self.index {
                if index.metric == metric {
                    let results = index.search(query, top_k, ef_search);
                    return to_scored_results(&results);
                }
            }
            return self.search_bruteforce(query, top_k, metric, None, None);
        }

        let filter_value = filters.unwrap();
        if let Some(filter_index) = &self.filters {
            if let Some(allowed) = filter_index.evaluate(filter_value) {
                if allowed.is_empty() {
                    return Vec::new();
                }
                if let Some(index) = &self.index {
                    if index.metric == metric {
                        let results = index.search_with_filter(query, top_k, ef_search, &allowed);
                        return to_scored_results(&results);
                    }
                }
                return self.search_bruteforce(query, top_k, metric, None, Some(&allowed));
            }
        }

        self.search_bruteforce(query, top_k, metric, Some(filter_value), None)
    }

    fn search_bruteforce(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
        filters: Option<&AttrValue>,
        allowed: Option<&RoaringBitmap>,
    ) -> Vec<ScoredResult> {
        let mut results = Vec::new();

        if let Some(bitmap) = allowed {
            for idx in bitmap.iter() {
                let idx = idx as usize;
                if idx >= self.vectors.len() {
                    continue;
                }
                if let Some(f) = filters {
                    if !matches_filters(self.attributes.get(idx).and_then(|v| v.as_ref()), f) {
                        continue;
                    }
                }
                let dist = distance(query, &self.vectors[idx], metric);
                results.push(ScoredResult { index: idx, dist });
            }
        } else {
            for (idx, vec) in self.vectors.iter().enumerate() {
                if let Some(f) = filters {
                    if !matches_filters(self.attributes.get(idx).and_then(|v| v.as_ref()), f) {
                        continue;
                    }
                }
                let dist = distance(query, vec, metric);
                results.push(ScoredResult { index: idx, dist });
            }
        }

        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        if results.len() > top_k {
            results.truncate(top_k);
        }
        results
    }
}

#[derive(Debug, Clone)]
pub struct FilterIndex {
    string_filters: std::collections::HashMap<String, std::collections::HashMap<String, RoaringBitmap>>,
}

fn build_filter_index(attributes: &[Option<AttrValue>]) -> Option<FilterIndex> {
    let mut string_filters: std::collections::HashMap<_, std::collections::HashMap<_, RoaringBitmap>> =
        std::collections::HashMap::new();

    for (doc_id, attrs) in attributes.iter().enumerate() {
        let Some(AttrValue::Object(map)) = attrs else { continue };
        for (key, value) in map {
            let Some(value_key) = filter_value_key(value) else { continue };
            let entry = string_filters.entry(key.clone()).or_default();
            let bitmap = entry.entry(value_key).or_insert_with(RoaringBitmap::new);
            bitmap.insert(doc_id as u32);
        }
    }

    if string_filters.is_empty() {
        None
    } else {
        Some(FilterIndex { string_filters })
    }
}

impl FilterIndex {
    pub fn evaluate(&self, filters: &AttrValue) -> Option<RoaringBitmap> {
        let AttrValue::Object(map) = filters else { return None };
        if map.is_empty() {
            return None;
        }

        let mut result: Option<RoaringBitmap> = None;
        for (key, value) in map {
            let field_map = self.string_filters.get(key)?;
            let mut field_bitmap = RoaringBitmap::new();
            let mut matched = false;

            match value {
                AttrValue::Array(items) => {
                    if items.is_empty() {
                        return Some(RoaringBitmap::new());
                    }
                    for item in items {
                        let Some(value_key) = filter_value_key(item) else { return None };
                        if let Some(bitmap) = field_map.get(&value_key) {
                            field_bitmap |= bitmap.clone();
                            matched = true;
                        }
                    }
                }
                _ => {
                    let Some(value_key) = filter_value_key(value) else { return None };
                    if let Some(bitmap) = field_map.get(&value_key) {
                        field_bitmap |= bitmap.clone();
                        matched = true;
                    }
                }
            }

            if !matched || field_bitmap.is_empty() {
                return Some(RoaringBitmap::new());
            }

            if let Some(mut current) = result {
                current &= field_bitmap;
                if current.is_empty() {
                    return Some(RoaringBitmap::new());
                }
                result = Some(current);
            } else {
                result = Some(field_bitmap);
            }
        }

        result
    }
}

fn filter_value_key(value: &AttrValue) -> Option<String> {
    match value {
        AttrValue::String(s) => Some(format!("s:{}", s)),
        AttrValue::Bool(b) => Some(format!("b:{}", if *b { 1 } else { 0 })),
        AttrValue::Number(num) => Some(format!("n:{}", num)),
        _ => None,
    }
}

fn matches_filters(attrs: Option<&AttrValue>, filters: &AttrValue) -> bool {
    let Some(AttrValue::Object(attr_map)) = attrs else { return false };
    let AttrValue::Object(filter_map) = filters else { return false };

    for (key, expected) in filter_map {
        let Some(actual) = attr_map.get(key) else { return false };
        match expected {
            AttrValue::Array(items) => {
                if !items.iter().any(|item| item == actual) {
                    return false;
                }
            }
            _ => {
                if actual != expected {
                    return false;
                }
            }
        }
    }

    true
}

fn to_scored_results(results: &[ResultItem]) -> Vec<ScoredResult> {
    results
        .iter()
        .map(|r| ScoredResult {
            index: r.id,
            dist: r.dist,
        })
        .collect()
}

fn segment_id_from_key(segment_key: &str) -> Option<String> {
    let filename = Path::new(segment_key).file_name()?.to_string_lossy();
    filename.strip_suffix(".tpvs").map(|s| s.to_string())
}

fn cache_key(key: &str, ext: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let hash = hasher.finalize();
    let hex = hex_encode(hash);
    format!("{}.{}", &hex[..16], ext)
}

async fn map_cached_file(path: String) -> Option<Mmap> {
    task::spawn_blocking(move || {
        let file = File::open(&path).ok()?;
        // SAFETY: the file is not mutated while mapped
        unsafe { Mmap::map(&file).ok() }
    })
    .await
    .ok()
    .flatten()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestSegment {
    pub id: String,
    pub segment_key: String,
    pub doc_count: i64,
    pub dimensions: usize,
}
