use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

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
use crate::segment_v2::{self, compute_norm, write_segment_v2, is_v2_format, SegmentView};
use crate::storage::{segment_index_path, segment_path, Store, StorageError};
use crate::vector::DistanceMetric;

/// Segment data - supports both owned (v1) and zero-copy (v2) modes
#[derive(Clone)]
pub struct SegmentData {
    /// String IDs (owned for v1, lazily loaded for v2)
    pub ids: Vec<String>,
    /// Vectors - for v1 these are owned, for v2 this is empty and we use raw_data
    pub vectors: Vec<Vec<f32>>,
    /// Attributes
    pub attributes: Vec<Option<AttrValue>>,
    /// Vector dimensions
    pub dimensions: usize,
    /// HNSW index for ANN search
    pub index: Option<HnswIndex>,
    /// Filter index for attribute queries
    pub filters: Option<FilterIndex>,
    /// Raw segment data for zero-copy v2 access (kept for lifetime)
    #[allow(dead_code)]
    raw_data: Option<Arc<SegmentBytes>>,
    /// Precomputed norms (for v2, stored in file; for v1, computed here)
    norms: Vec<f32>,
    /// Whether this is v2 format
    #[allow(dead_code)]
    is_v2: bool,
}

impl std::fmt::Debug for SegmentData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentData")
            .field("ids_len", &self.ids.len())
            .field("vectors_len", &self.vectors.len())
            .field("dimensions", &self.dimensions)
            .field("has_index", &self.index.is_some())
            .field("is_v2", &self.is_v2)
            .finish()
    }
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
    /// Use v2 zero-copy format (default: true)
    pub use_v2_format: bool,
}

impl Default for WriterOptions {
    fn default() -> Self {
        Self {
            hnsw_m: DEFAULT_M,
            hnsw_ef_construction: DEFAULT_EF_CONSTRUCTION,
            hnsw_ef_search: DEFAULT_EF_SEARCH,
            metric: DistanceMetric::Cosine,
            use_v2_format: true,
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
        
        // Write segment data (v2 or v1 format)
        let buf = if self.opts.use_v2_format {
            write_segment_v2(docs)?
        } else {
            self.write_segment_v1(docs, dimensions)?
        };

        // Build HNSW index
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
    
    /// Write v1 format segment (legacy)
    fn write_segment_v1(&self, docs: &[Document], dimensions: usize) -> Result<Vec<u8>, StorageError> {
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
        buf.extend_from_slice(attr_bytes);
        
        Ok(buf)
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
        let bytes = data.as_slice();
        
        // Check format version
        if is_v2_format(bytes) {
            self.read_segment_v2(segment_key, data).await
        } else {
            self.read_segment_v1(segment_key, data).await
        }
    }
    
    /// Read v2 format segment (zero-copy)
    async fn read_segment_v2(&self, segment_key: &str, data: SegmentBytes) -> Result<SegmentData, StorageError> {
        let data = Arc::new(data);
        let bytes = data.as_slice();
        
        // SAFETY: We verify the format and bounds in SegmentView::from_bytes
        let view = unsafe { SegmentView::from_bytes(bytes)? };
        
        let num_vectors = view.len();
        let dimensions = view.dimensions();
        
        // Extract IDs (we need owned strings for the API)
        let mut ids = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            ids.push(view.id_string(i).unwrap_or("").to_string());
        }
        
        // Extract vectors (needed for HNSW which stores references)
        // TODO: In future, make HNSW work with zero-copy views
        let mut vectors = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            vectors.push(view.vector(i).to_vec());
        }
        
        // Extract norms
        let mut norms = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            norms.push(view.norm(i));
        }
        
        // Extract attributes
        let (attr_bytes, attr_len) = view.attr_region();
        let attrs = if attr_len > 0 {
            // SAFETY: We trust our own serialized format
            let archived = unsafe { rkyv::archived_root::<Vec<segment_v2::SegmentAttrV2>>(attr_bytes) };
            let attr_data: Vec<segment_v2::SegmentAttrV2> = archived
                .deserialize(&mut rkyv::Infallible)
                .map_err(|e| StorageError::Other(format!("deserialize v2 attributes: {:?}", e)))?;
            
            attr_data
                .into_iter()
                .map(|a| {
                    if a.attributes_json.is_empty() {
                        None
                    } else {
                        serde_json::from_slice(&a.attributes_json).ok()
                    }
                })
                .collect()
        } else {
            vec![None; num_vectors]
        };
        
        let mut seg = SegmentData {
            ids,
            vectors,
            attributes: attrs,
            dimensions,
            index: None,
            filters: None,
            raw_data: Some(data),
            norms,
            is_v2: true,
        };
        
        // Load HNSW index
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
    
    /// Read v1 format segment (legacy)
    async fn read_segment_v1(&self, segment_key: &str, data: SegmentBytes) -> Result<SegmentData, StorageError> {
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
        let mut norms = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let mut vec = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                let v = cursor
                    .read_f32::<LittleEndian>()
                    .map_err(|e| StorageError::Other(format!("read vector: {}", e)))?;
                vec.push(v);
            }
            norms.push(compute_norm(&vec));
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
            raw_data: None,
            norms,
            is_v2: false,
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

    /// Remove cached files for segments not in the valid set.
    /// Call this after loading a new manifest to clean up stale cache entries.
    /// Returns the number of files removed.
    pub async fn cleanup_cache(&self, valid_segment_keys: &[String]) -> usize {
        let Some(dir) = &self.cache_dir else { return 0 };

        // Build set of valid cache file prefixes (hash of segment key)
        let valid_prefixes: std::collections::HashSet<String> = valid_segment_keys
            .iter()
            .map(|key| cache_key_prefix(key))
            .collect();

        // Scan cache directory and remove stale files
        let read_dir = match std::fs::read_dir(dir) {
            Ok(rd) => rd,
            Err(_) => return 0,
        };

        let mut removed = 0usize;
        for entry in read_dir.flatten() {
            let path = entry.path();
            let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };

            // Only process our cache files (.tpvs, .hnsw)
            if !filename.ends_with(".tpvs") && !filename.ends_with(".hnsw") {
                continue;
            }

            // Extract the hash prefix (first 16 chars before the extension)
            let prefix = filename.split('.').next().unwrap_or("");
            if prefix.len() != 16 {
                continue;
            }

            // If prefix not in valid set, remove the file
            if !valid_prefixes.contains(prefix) {
                if std::fs::remove_file(&path).is_ok() {
                    removed += 1;
                }
            }
        }

        removed
    }
}

/// Get just the hash prefix for a cache key (for comparison during cleanup)
fn cache_key_prefix(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let hash = hasher.finalize();
    let hex = hex_encode(hash);
    hex[..16].to_string()
}

/// Segment data storage - either owned or memory-mapped
pub enum SegmentBytes {
    Owned(Vec<u8>),
    Mapped(Mmap),
}

impl Clone for SegmentBytes {
    fn clone(&self) -> Self {
        match self {
            SegmentBytes::Owned(buf) => SegmentBytes::Owned(buf.clone()),
            // For Mmap, we just clone the data (mmap cannot be cloned)
            SegmentBytes::Mapped(mmap) => SegmentBytes::Owned(mmap.to_vec()),
        }
    }
}

impl SegmentBytes {
    pub fn as_slice(&self) -> &[u8] {
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
    /// Get the number of vectors
    #[inline]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    
    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    
    /// Get vector by index
    #[inline]
    pub fn vector(&self, index: usize) -> &[f32] {
        &self.vectors[index]
    }
    
    /// Get precomputed norm for vector
    #[inline]
    pub fn norm(&self, index: usize) -> f32 {
        if index < self.norms.len() {
            self.norms[index]
        } else {
            compute_norm(&self.vectors[index])
        }
    }
    
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
        filters: Option<&AttrValue>,
        ef_search: usize,
    ) -> Vec<ScoredResult> {
        let top_k = if top_k == 0 { self.vectors.len() } else { top_k };

        // Require HNSW index for search
        let Some(index) = &self.index else {
            return Vec::new();
        };

        // Use index metric (ignore requested metric if different)
        let _ = metric;

        if filters.is_none() {
            let results = index.search(query, top_k, ef_search);
            return to_scored_results(&results);
        }

        let filter_value = filters.unwrap();
        if let Some(filter_index) = &self.filters {
            if let Some(allowed) = filter_index.evaluate(filter_value) {
                if allowed.is_empty() {
                    return Vec::new();
                }
                let results = index.search_with_filter(query, top_k, ef_search, &allowed);
                return to_scored_results(&results);
            }
        }

        // Filter couldn't be evaluated by index, return empty
        // (In production, this would fall back to post-filtering)
        Vec::new()
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
