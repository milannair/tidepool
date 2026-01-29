use std::collections::BinaryHeap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use hex::encode as hex_encode;
use memmap2::Mmap;
use ordered_float::OrderedFloat;
use roaring::RoaringBitmap;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::task;
use tokio::fs;

use crate::attributes::AttrValue;
use crate::document::Document;
use crate::index::hnsw::{HnswIndex, ResultItem, DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH, DEFAULT_M};
use crate::index::ivf::IVFIndex;
use crate::index::text::TextIndex;
use crate::quantization::{self, QuantizationKind, QuantizedVectors, Sq8Query};
use crate::segment_v2::{self, compute_norm, write_segment_v2, write_segment_v3, is_v2_format, is_v3_format, SegmentLayout, SegmentView};
use crate::storage::{
    segment_index_path,
    segment_ivf_path,
    segment_path,
    segment_quant_path,
    segment_text_index_path,
    Store,
    StorageError,
};
use crate::text::DefaultTokenizer;
use crate::vector::{distance_with_norms, DistanceMetric};

/// Segment data - supports both owned (v1) and zero-copy (v2) modes
#[derive(Clone)]
pub struct SegmentData {
    /// String IDs (owned for v1, lazily loaded for v2)
    pub ids: Vec<String>,
    /// Vectors - for v1 these are owned, for v2 this is empty and we use raw_data
    pub vectors: Vec<Vec<f32>>,
    /// Attributes
    pub attributes: Vec<Option<AttrValue>>,
    /// Stored text (v3 segments)
    pub texts: Vec<Option<String>>,
    /// Vector dimensions
    pub dimensions: usize,
    /// HNSW index for ANN search
    pub index: Option<HnswIndex>,
    /// IVF index for ANN search (large segments)
    pub ivf_index: Option<IVFIndex>,
    /// Quantized vectors for IVF search
    pub quantization: Option<QuantizedVectors>,
    /// Filter index for attribute queries
    pub filters: Option<FilterIndex>,
    /// Text index for BM25 search
    pub text_index: Option<TextIndex>,
    /// Raw segment data for zero-copy v2 access (kept for lifetime)
    #[allow(dead_code)]
    raw_data: Option<Arc<SegmentBytes>>,
    /// Parsed layout for v2 segments (used for on-demand vector access)
    raw_layout: Option<SegmentLayout>,
    /// Precomputed norms (for v2, stored in file; for v1, computed here)
    norms: Vec<f32>,
    /// Whether this is v2 format
    #[allow(dead_code)]
    is_v2: bool,
    /// Total vector count (may differ from vectors.len when quantized)
    vector_count: usize,
    /// Quantized rerank factor for IVF (>=1)
    quantization_rerank_factor: usize,
}

impl std::fmt::Debug for SegmentData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentData")
            .field("ids_len", &self.ids.len())
            .field("vectors_len", &self.vectors.len())
            .field("vector_count", &self.vector_count)
            .field("dimensions", &self.dimensions)
            .field("has_index", &self.index.is_some())
            .field("has_ivf", &self.ivf_index.is_some())
            .field("has_quant", &self.quantization.is_some())
            .field("has_text_index", &self.text_index.is_some())
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
    /// Use v3 segment format (default: true)
    pub use_v3_format: bool,
    /// Enable text index build
    pub text_index_enabled: bool,
    /// Tokenizer config for text index
    pub tokenizer_config: crate::text::TokenizerConfig,
    /// Enable IVF index build
    pub ivf_enabled: bool,
    /// Minimum vectors required to build IVF (otherwise HNSW)
    pub ivf_min_segment_size: usize,
    /// IVF k scaling factor (k ≈ sqrt(n) * factor)
    pub ivf_k_factor: f32,
    /// IVF k lower bound
    pub ivf_min_k: usize,
    /// IVF k upper bound
    pub ivf_max_k: usize,
    /// IVF default nprobe (stored in index)
    pub ivf_nprobe_default: usize,
    /// IVF k-means iterations
    pub ivf_max_iters: usize,
    /// IVF deterministic seed
    pub ivf_seed: u64,
    /// Quantization mode for IVF segments
    pub quantization: QuantizationKind,
}

impl Default for WriterOptions {
    fn default() -> Self {
        Self {
            hnsw_m: DEFAULT_M,
            hnsw_ef_construction: DEFAULT_EF_CONSTRUCTION,
            hnsw_ef_search: DEFAULT_EF_SEARCH,
            metric: DistanceMetric::Cosine,
            use_v3_format: true,
            text_index_enabled: true,
            tokenizer_config: crate::text::TokenizerConfig::default(),
            ivf_enabled: true,
            ivf_min_segment_size: 10_000,
            ivf_k_factor: 1.0,
            ivf_min_k: 16,
            ivf_max_k: 65_535,
            ivf_nprobe_default: 10,
            ivf_max_iters: 25,
            ivf_seed: 42,
            quantization: QuantizationKind::None,
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
        
        // Write segment data (v3 or v2 format)
        let buf = if self.opts.use_v3_format {
            write_segment_v3(docs)?
        } else {
            write_segment_v2(docs)?
        };

        let segment_key = segment_path(&self.namespace, &segment_id);
        self.storage.put(&segment_key, buf).await?;

        let use_ivf = self.opts.ivf_enabled && docs.len() >= self.opts.ivf_min_segment_size;
        let mut quant_key: Option<String> = None;
        if self.opts.use_v3_format
            && use_ivf
            && self.opts.quantization != QuantizationKind::None
        {
            let vector_refs: Vec<&[f32]> = docs.iter().map(|d| d.vector.as_slice()).collect();
            let quant = quantization::quantize(&vector_refs, self.opts.quantization)
                .map_err(|e| StorageError::Other(format!("quantize vectors: {}", e)))?;
            let quant_data = quantization::marshal_binary(&quant)
                .map_err(|e| StorageError::Other(format!("serialize quantization: {}", e)))?;
            let key = segment_quant_path(&self.namespace, &segment_id);
            if let Err(err) = self.storage.put(&key, quant_data).await {
                let _ = self.storage.delete(&segment_key).await;
                return Err(err);
            }
            quant_key = Some(key);
        }

        if use_ivf {
            let vector_refs: Vec<&[f32]> = docs.iter().map(|d| d.vector.as_slice()).collect();
            let k = IVFIndex::compute_k(
                vector_refs.len(),
                self.opts.ivf_k_factor,
                self.opts.ivf_min_k,
                self.opts.ivf_max_k,
            );
            let ivf = IVFIndex::build(
                &vector_refs,
                self.opts.metric,
                k,
                self.opts.ivf_nprobe_default,
                self.opts.ivf_max_iters,
                self.opts.ivf_seed,
            )
            .map_err(|e| StorageError::Other(format!("build IVF: {}", e)))?;
            let ivf_data = ivf
                .marshal_binary()
                .map_err(|e| StorageError::Other(format!("serialize IVF: {}", e)))?;
            let ivf_key = segment_ivf_path(&self.namespace, &segment_id);
            if let Err(err) = self.storage.put(&ivf_key, ivf_data).await {
                let _ = self.storage.delete(&segment_key).await;
                if let Some(quant_key) = &quant_key {
                    let _ = self.storage.delete(quant_key).await;
                }
                return Err(err);
            }
        } else {
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

            let index_key = segment_index_path(&self.namespace, &segment_id);
            if let Err(err) = self.storage.put(&index_key, hnsw_data).await {
                let _ = self.storage.delete(&segment_key).await;
                if let Some(quant_key) = &quant_key {
                    let _ = self.storage.delete(quant_key).await;
                }
                return Err(err);
            }
        }

        // Build text index (optional)
        if self.opts.text_index_enabled {
            let tokenizer = DefaultTokenizer::new(self.opts.tokenizer_config.clone());
            if let Some(text_index) = TextIndex::build(docs, &tokenizer) {
                let text_data = text_index
                    .marshal_binary()
                    .map_err(|e| StorageError::Other(format!("serialize text index: {}", e)))?;
                let text_key = segment_text_index_path(&self.namespace, &segment_id);
                if let Err(err) = self.storage.put(&text_key, text_data).await {
                    let _ = self.storage.delete(&segment_key).await;
                    let _ = self.storage.delete(&segment_index_path(&self.namespace, &segment_id)).await;
                    let _ = self.storage.delete(&segment_ivf_path(&self.namespace, &segment_id)).await;
                    let _ = self.storage.delete(&segment_quant_path(&self.namespace, &segment_id)).await;
                    return Err(err);
                }
            }
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
    pub quantization_rerank_factor: usize,
}

impl Default for ReaderOptions {
    fn default() -> Self {
        Self {
            hnsw_ef_search: DEFAULT_EF_SEARCH,
            quantization_rerank_factor: 4,
        }
    }
}

#[derive(Clone)]
pub struct Reader<S: Store> {
    storage: S,
    namespace: String,
    cache_dir: Option<String>,
    hnsw_ef_search: usize,
    quantization_rerank_factor: usize,
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
            quantization_rerank_factor: opts.quantization_rerank_factor.max(1),
        }
    }

    pub async fn read_segment(&self, segment_key: &str) -> Result<SegmentData, StorageError> {
        let data = self.get_segment_data(segment_key).await?;
        let bytes = data.as_slice();
        
        // Check format version
        if is_v2_format(bytes) || is_v3_format(bytes) {
            self.read_segment_v2(segment_key, data).await
        } else {
            self.read_segment_v1(segment_key, data).await
        }
    }
    
    /// Read v2 format segment (zero-copy)
    async fn read_segment_v2(&self, segment_key: &str, data: SegmentBytes) -> Result<SegmentData, StorageError> {
        let data = Arc::new(data);
        let bytes = data.as_slice();
        
        let layout = SegmentLayout::parse(bytes)?;
        let is_v2 = is_v2_format(bytes);
        // SAFETY: We verify the format and bounds in SegmentView::from_bytes
        let view = unsafe { SegmentView::from_bytes(bytes)? };
        
        let num_vectors = view.len();
        let dimensions = view.dimensions();
        
        // Extract IDs (we need owned strings for the API)
        let mut ids = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            ids.push(view.id_string(i).unwrap_or("").to_string());
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

        // Extract text table (v3 only)
        let texts = if layout.text_len > 0 {
            parse_text_table(bytes, layout.text_offset, layout.text_len, num_vectors)?
        } else {
            Vec::new()
        };
        
        // Load IVF index + quantization sidecar (if present)
        let mut ivf_index = None;
        let mut quantization = None;
        if let Some(segment_id) = segment_id_from_key(segment_key) {
            let ivf_key = segment_ivf_path(&self.namespace, &segment_id);
            if self.storage.exists(&ivf_key).await.unwrap_or(false) {
                if let Ok(ivf_data) = self.get_ivf_data(&ivf_key).await {
                    if let Ok(ivf) = IVFIndex::load_binary(ivf_data.as_slice()) {
                        ivf_index = Some(ivf);
                    }
                }
            }
            let quant_key = segment_quant_path(&self.namespace, &segment_id);
            if self.storage.exists(&quant_key).await.unwrap_or(false) {
                if let Ok(quant_data) = self.get_quant_data(&quant_key).await {
                    if let Ok(q) = quantization::load_binary(quant_data.as_slice()) {
                        if q.dimensions == dimensions && q.vector_count == num_vectors {
                            quantization = Some(q);
                        }
                    }
                }
            }
        }

        // Extract vectors if needed (before moving data into struct)
        let need_vectors = ivf_index.is_none() || quantization.is_none();
        let vectors = if need_vectors {
            // Extract vectors (needed for HNSW or exact IVF scan)
            let mut vecs = Vec::with_capacity(num_vectors);
            for i in 0..num_vectors {
                vecs.push(view.vector(i).to_vec());
            }
            vecs
        } else {
            Vec::new()
        };

        let mut seg = SegmentData {
            ids,
            vectors,
            attributes: attrs,
            texts,
            dimensions,
            index: None,
            ivf_index,
            quantization,
            filters: None,
            text_index: None,
            raw_data: Some(data),
            raw_layout: Some(layout),
            norms,
            is_v2,
            vector_count: num_vectors,
            quantization_rerank_factor: self.quantization_rerank_factor,
        };

        // Load text index sidecar (if present)
        if let Some(segment_id) = segment_id_from_key(segment_key) {
            let text_key = segment_text_index_path(&self.namespace, &segment_id);
            if self.storage.exists(&text_key).await.unwrap_or(false) {
                if let Ok(text_data) = self.get_text_index_data(&text_key).await {
                    if let Ok(idx) = TextIndex::load_binary(text_data.as_slice()) {
                        seg.text_index = Some(idx);
                    }
                }
            }
        }

        // Load HNSW index only when IVF is not present
        if seg.ivf_index.is_none() {
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
            texts: Vec::new(),
            dimensions,
            index: None,
            ivf_index: None,
            quantization: None,
            filters: None,
            text_index: None,
            raw_data: None,
            raw_layout: None,
            norms,
            is_v2: false,
            vector_count: num_vectors,
            quantization_rerank_factor: self.quantization_rerank_factor,
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
            let ivf_key = segment_ivf_path(&self.namespace, &segment_id);
            if self.storage.exists(&ivf_key).await.unwrap_or(false) {
                if let Ok(ivf_data) = self.get_ivf_data(&ivf_key).await {
                    if let Ok(ivf) = IVFIndex::load_binary(ivf_data.as_slice()) {
                        seg.ivf_index = Some(ivf);
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

    async fn get_ivf_data(&self, ivf_key: &str) -> Result<SegmentBytes, StorageError> {
        if let Some(dir) = &self.cache_dir {
            let cache_path = format!("{}/{}", dir, cache_key(ivf_key, "ivf"));
            if let Some(mmap) = map_cached_file(cache_path.clone()).await {
                return Ok(SegmentBytes::Mapped(mmap));
            }
            if let Ok(data) = fs::read(&cache_path).await {
                return Ok(SegmentBytes::Owned(data));
            }
        }
        let data = self.storage.get(ivf_key).await?;
        if let Some(dir) = &self.cache_dir {
            let cache_path = format!("{}/{}", dir, cache_key(ivf_key, "ivf"));
            let _ = fs::write(cache_path, &data).await;
        }
        Ok(SegmentBytes::Owned(data))
    }

    async fn get_quant_data(&self, quant_key: &str) -> Result<SegmentBytes, StorageError> {
        if let Some(dir) = &self.cache_dir {
            let cache_path = format!("{}/{}", dir, cache_key(quant_key, "tpq"));
            if let Some(mmap) = map_cached_file(cache_path.clone()).await {
                return Ok(SegmentBytes::Mapped(mmap));
            }
            if let Ok(data) = fs::read(&cache_path).await {
                return Ok(SegmentBytes::Owned(data));
            }
        }
        let data = self.storage.get(quant_key).await?;
        if let Some(dir) = &self.cache_dir {
            let cache_path = format!("{}/{}", dir, cache_key(quant_key, "tpq"));
            let _ = fs::write(cache_path, &data).await;
        }
        Ok(SegmentBytes::Owned(data))
    }

    async fn get_text_index_data(&self, text_key: &str) -> Result<SegmentBytes, StorageError> {
        if let Some(dir) = &self.cache_dir {
            let cache_path = format!("{}/{}", dir, cache_key(text_key, "tpti"));
            if let Some(mmap) = map_cached_file(cache_path.clone()).await {
                return Ok(SegmentBytes::Mapped(mmap));
            }
            if let Ok(data) = fs::read(&cache_path).await {
                return Ok(SegmentBytes::Owned(data));
            }
        }
        let data = self.storage.get(text_key).await?;
        if let Some(dir) = &self.cache_dir {
            let cache_path = format!("{}/{}", dir, cache_key(text_key, "tpti"));
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

            // Only process our cache files (.tpvs, .hnsw, .ivf, .tpq)
            if !filename.ends_with(".tpvs")
                && !filename.ends_with(".hnsw")
                && !filename.ends_with(".ivf")
                && !filename.ends_with(".tpq")
                && !filename.ends_with(".tpti")
            {
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

fn parse_text_table(
    bytes: &[u8],
    offset: usize,
    len: usize,
    count: usize,
) -> Result<Vec<Option<String>>, StorageError> {
    if len == 0 || count == 0 {
        return Ok(Vec::new());
    }
    if offset + len > bytes.len() {
        return Err(StorageError::Other("text table out of bounds".into()));
    }

    let mut out = Vec::with_capacity(count);
    let mut cursor = std::io::Cursor::new(&bytes[offset..offset + len]);
    for _ in 0..count {
        if (cursor.position() as usize) >= len {
            out.push(None);
            continue;
        }
        let str_len = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| StorageError::Other(format!("read text length: {}", e)))? as usize;
        if str_len == 0 {
            out.push(None);
            continue;
        }
        let mut buf = vec![0u8; str_len];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| StorageError::Other(format!("read text: {}", e)))?;
        let text = String::from_utf8(buf)
            .map_err(|e| StorageError::Other(format!("decode text: {}", e)))?;
        out.push(Some(text));
    }

    Ok(out)
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

#[derive(Debug, Clone)]
pub struct TextScoredResult {
    pub index: usize,
    pub score: f32,
}

impl SegmentData {
    /// Get the number of vectors
    #[inline]
    pub fn len(&self) -> usize {
        self.vector_count
    }
    
    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vector_count == 0
    }
    
    /// Get vector by index
    #[inline]
    pub fn vector(&self, index: usize) -> &[f32] {
        self.vector_slice(index).expect("vector not available")
    }

    /// Get vector slice by index (owned vectors or mmap-backed)
    #[inline]
    pub fn vector_slice(&self, index: usize) -> Option<&[f32]> {
        if index >= self.vector_count {
            return None;
        }
        if !self.vectors.is_empty() {
            return self.vectors.get(index).map(|v| v.as_slice());
        }
        let layout = self.raw_layout?;
        let data = self.raw_data.as_ref()?;
        let dims = layout.dimensions;
        let start = layout.vector_offset + index * dims * std::mem::size_of::<f32>();
        let end = start + dims * std::mem::size_of::<f32>();
        let bytes = data.as_slice();
        if end > bytes.len() {
            return None;
        }
        unsafe {
            Some(std::slice::from_raw_parts(
                bytes.as_ptr().add(start) as *const f32,
                dims,
            ))
        }
    }

    /// Get vector by index as an owned Vec (for API responses)
    #[inline]
    pub fn vector_owned(&self, index: usize) -> Option<Vec<f32>> {
        self.vector_slice(index).map(|v| v.to_vec())
    }
    
    /// Get precomputed norm for vector
    #[inline]
    pub fn norm(&self, index: usize) -> f32 {
        if index < self.norms.len() {
            self.norms[index]
        } else {
            self.vector_slice(index)
                .map(compute_norm)
                .unwrap_or(0.0)
        }
    }
    
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
        filters: Option<&AttrValue>,
        ef_search: usize,
        nprobe: usize,
    ) -> Vec<ScoredResult> {
        if query.is_empty() || query.len() != self.dimensions {
            return Vec::new();
        }
        let top_k = if top_k == 0 { self.len() } else { top_k };

        if let Some(ivf) = &self.ivf_index {
            return self.ivf_search(query, top_k, ivf.metric, filters, nprobe);
        }

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

    pub fn text_search(
        &self,
        tokens: &[String],
        top_k: usize,
        filters: Option<&AttrValue>,
        bm25_k1: f32,
        bm25_b: f32,
    ) -> Vec<TextScoredResult> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let Some(index) = &self.text_index else {
            return Vec::new();
        };

        let mut allowed = None;
        if let Some(filter_value) = filters {
            if let Some(filter_index) = &self.filters {
                allowed = filter_index.evaluate(filter_value);
                if let Some(bitmap) = &allowed {
                    if bitmap.is_empty() {
                        return Vec::new();
                    }
                }
            } else {
                return Vec::new();
            }
        }

        let results = index.search(tokens, top_k, allowed.as_ref(), bm25_k1, bm25_b);
        results
            .into_iter()
            .map(|(doc_id, score)| TextScoredResult {
                index: doc_id as usize,
                score,
            })
            .collect()
    }

    fn ivf_search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
        filters: Option<&AttrValue>,
        nprobe: usize,
    ) -> Vec<ScoredResult> {
        let Some(ivf) = &self.ivf_index else {
            return Vec::new();
        };

        let top_k = top_k.min(self.len());
        if top_k == 0 || query.is_empty() {
            return Vec::new();
        }

        let mut allowed = None;
        if let Some(filter_value) = filters {
            if let Some(filter_index) = &self.filters {
                allowed = filter_index.evaluate(filter_value);
                if let Some(bitmap) = &allowed {
                    if bitmap.is_empty() {
                        return Vec::new();
                    }
                }
            } else {
                return Vec::new();
            }
        }

        let query_norm = if metric == DistanceMetric::Cosine {
            crate::simd::l2_norm_f32(query)
        } else {
            0.0
        };

        let mut centroid_scores: Vec<(usize, f32)> = Vec::with_capacity(ivf.k);
        for c in 0..ivf.k {
            let dist = distance_with_norms(
                query,
                ivf.centroid(c),
                query_norm,
                ivf.centroid_norm(c),
                metric,
            );
            centroid_scores.push((c, dist));
        }
        centroid_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut nprobe = if nprobe == 0 { ivf.nprobe_default } else { nprobe };
        nprobe = nprobe.clamp(1, ivf.k);
        centroid_scores.truncate(nprobe);

        let use_quant = self.quantization.as_ref();
        let rerank_factor = self.quantization_rerank_factor.max(1);
        let target_k = if use_quant.is_some() {
            (top_k * rerank_factor).min(self.len())
        } else {
            top_k
        };

        let mut sq8_query: Option<Sq8Query> = None;
        if let Some(q) = use_quant {
            if q.kind == QuantizationKind::SQ8 {
                sq8_query = Some(Sq8Query::new(query, &q.scales, &q.mins, query_norm));
            }
        }

        let mut heap: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        for (c, centroid_dist) in centroid_scores {
            if metric == DistanceMetric::Euclidean && heap.len() >= target_k && use_quant.is_none() {
                let worst = heap.peek().map(|v| v.0.into_inner()).unwrap_or(f32::INFINITY);
                let worst_l2 = worst.sqrt();
                let centroid_l2 = centroid_dist.sqrt();
                if centroid_l2 - ivf.radii[c] > worst_l2 {
                    continue;
                }
            }

            for &vid in &ivf.posting_lists[c] {
                if let Some(bitmap) = &allowed {
                    if !bitmap.contains(vid) {
                        continue;
                    }
                }
                let vid = vid as usize;
                if vid >= self.len() {
                    continue;
                }
                let dist = if let Some(q) = use_quant {
                    match q.kind {
                        QuantizationKind::F16 => q
                            .vector_bytes(vid)
                            .map(|bytes| {
                                quantization::f16_distance(
                                    query,
                                    bytes,
                                    metric,
                                    query_norm,
                                    self.norm(vid),
                                )
                            })
                            .or_else(|| {
                                self.vector_slice(vid).map(|vec| {
                                    distance_with_norms(
                                        query,
                                        vec,
                                        query_norm,
                                        self.norm(vid),
                                        metric,
                                    )
                                })
                            }),
                        QuantizationKind::SQ8 => q
                            .vector_bytes(vid)
                            .and_then(|bytes| sq8_query.as_ref().map(|sq8| (bytes, sq8)))
                            .map(|(bytes, sq8)| {
                                quantization::sq8_distance(
                                    sq8,
                                    bytes,
                                    &q.scales,
                                    &q.mins,
                                    metric,
                                    self.norm(vid),
                                )
                            })
                            .or_else(|| {
                                self.vector_slice(vid).map(|vec| {
                                    distance_with_norms(
                                        query,
                                        vec,
                                        query_norm,
                                        self.norm(vid),
                                        metric,
                                    )
                                })
                            }),
                        QuantizationKind::None => None,
                    }
                } else {
                    self.vector_slice(vid).map(|vec| {
                        distance_with_norms(query, vec, query_norm, self.norm(vid), metric)
                    })
                };

                let Some(dist) = dist else { continue };

                if heap.len() < target_k {
                    heap.push((OrderedFloat(dist), vid));
                } else if let Some(mut worst) = heap.peek_mut() {
                    if dist < worst.0.into_inner() {
                        *worst = (OrderedFloat(dist), vid);
                    }
                }
            }
        }

        if use_quant.is_some() {
            let mut candidates = Vec::with_capacity(heap.len());
            while let Some((dist, index)) = heap.pop() {
                candidates.push((index, dist.into_inner()));
            }

            let mut rerank_heap: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
            for (index, approx) in candidates {
                let dist = self
                    .vector_slice(index)
                    .map(|vec| {
                        distance_with_norms(query, vec, query_norm, self.norm(index), metric)
                    })
                    .unwrap_or(approx);

                if rerank_heap.len() < top_k {
                    rerank_heap.push((OrderedFloat(dist), index));
                } else if let Some(mut worst) = rerank_heap.peek_mut() {
                    if dist < worst.0.into_inner() {
                        *worst = (OrderedFloat(dist), index);
                    }
                }
            }

            let mut results = Vec::with_capacity(rerank_heap.len());
            while let Some((dist, index)) = rerank_heap.pop() {
                results.push(ScoredResult {
                    index,
                    dist: dist.into_inner(),
                });
            }
            results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            results
        } else {
            let mut results = Vec::with_capacity(heap.len());
            while let Some((dist, index)) = heap.pop() {
                results.push(ScoredResult {
                    index,
                    dist: dist.into_inner(),
                });
            }
            results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            results
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn exact_top_k(
        vectors: &[Vec<f32>],
        norms: &[f32],
        query: &[f32],
        query_norm: f32,
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<ScoredResult> {
        let mut scored: Vec<ScoredResult> = vectors
            .iter()
            .enumerate()
            .map(|(idx, v)| ScoredResult {
                index: idx,
                dist: distance_with_norms(query, v, query_norm, norms[idx], metric),
            })
            .collect();
        scored.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        scored.truncate(k);
        scored
    }

    #[test]
    fn ivf_search_matches_exact_with_full_probe() {
        let mut rng = StdRng::seed_from_u64(42);
        let dims = 8;
        let count = 64;
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|_| (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let norms: Vec<f32> = vectors.iter().map(|v| compute_norm(v)).collect();
        let ids: Vec<String> = (0..count).map(|i| format!("id{}", i)).collect();

        let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let k = 8;
        let ivf = IVFIndex::build(
            &vector_refs,
            DistanceMetric::Cosine,
            k,
            k,
            10,
            42,
        )
        .expect("ivf build");

        let seg = SegmentData {
            ids,
            vectors: vectors.clone(),
            attributes: vec![None; count],
            texts: Vec::new(),
            dimensions: dims,
            index: None,
            ivf_index: Some(ivf),
            quantization: None,
            filters: None,
            text_index: None,
            raw_data: None,
            raw_layout: None,
            norms,
            is_v2: false,
            vector_count: count,
            quantization_rerank_factor: 1,
        };

        let query = vectors[0].clone();
        let query_norm = crate::simd::l2_norm_f32(&query);
        let exact = exact_top_k(&vectors, &seg.norms, &query, query_norm, 5, DistanceMetric::Cosine);
        let results = seg.search(&query, 5, DistanceMetric::Cosine, None, 0, 0);

        let exact_ids: Vec<usize> = exact.iter().map(|r| r.index).collect();
        let result_ids: Vec<usize> = results.iter().map(|r| r.index).collect();
        assert_eq!(exact_ids, result_ids);
    }
}
