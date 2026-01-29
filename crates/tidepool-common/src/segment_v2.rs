//! Zero-copy segment format v2
//!
//! Layout:
//! ```text
//! [Header: 68 bytes (v2) / 84 bytes (v3)]
//!   magic: [u8; 4] = "TPV2" | "TPV3"
//!   version: u32 = 2 | 3
//!   vector_count: u32
//!   dimensions: u32
//!   flags: u32
//!     bit 0: vectors are pre-normalized
//!   norm_offset: u64
//!   vector_offset: u64
//!   id_offset: u64
//!   string_table_offset: u64
//!   attr_offset: u64
//!   attr_len: u64
//!   text_offset: u64 (v3 only)
//!   text_len: u64 (v3 only)
//! [Norms: f32 × n, 32-byte aligned]
//! [Vectors: f32 × n × d, 32-byte aligned]
//! [IDs: u64 × n]
//! [String table: length-prefixed strings]
//! [Attributes: rkyv archived]
//! [Text table: length-prefixed strings] (v3 only)
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::mem;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

use crate::document::Document;
use crate::simd;
use crate::storage::StorageError;

// Test-only imports
#[cfg(test)]
use roaring::RoaringBitmap;
#[cfg(test)]
use crate::vector::DistanceMetric;

/// Alignment for vector data (AVX2 friendly)
const ALIGNMENT: usize = 32;

/// Segment header sizes (must match actual written size)
const HEADER_SIZE_V2: usize = 68;
const HEADER_SIZE_V3: usize = 84;

/// Magic bytes for v2 format
const MAGIC_V2: &[u8; 4] = b"TPV2";
/// Magic bytes for v3 format
const MAGIC_V3: &[u8; 4] = b"TPV3";

/// Segment flags
pub mod flags {
    /// Vectors are pre-normalized to unit length
    /// When set, cosine distance can use: 1 - dot(a, b)
    pub const VECTORS_NORMALIZED: u32 = 1 << 0;
}

/// Zero-copy view over a segment file
pub struct SegmentView<'a> {
    data: &'a [u8],
    vector_count: usize,
    dimensions: usize,
    flags: u32,
    attr_offset: usize,
    attr_len: usize,
    norms: &'a [f32],
    vectors: &'a [f32],
    ids: &'a [u64],
    string_table: StringTableView<'a>,
}

/// Parsed header for segment v2
#[derive(Debug, Clone, Copy)]
pub struct SegmentHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub vector_count: u32,
    pub dimensions: u32,
    pub flags: u32,
    pub norm_offset: u64,
    pub vector_offset: u64,
    pub id_offset: u64,
    pub string_table_offset: u64,
    pub attr_offset: u64,
    pub attr_len: u64,
    pub text_offset: u64,
    pub text_len: u64,
}

/// Parsed layout for segment v2 (offsets in bytes)
#[derive(Debug, Clone, Copy)]
pub struct SegmentLayout {
    pub vector_count: usize,
    pub dimensions: usize,
    pub flags: u32,
    pub norm_offset: usize,
    pub vector_offset: usize,
    pub id_offset: usize,
    pub string_table_offset: usize,
    pub attr_offset: usize,
    pub attr_len: usize,
    pub text_offset: usize,
    pub text_len: usize,
}

impl SegmentLayout {
    pub fn parse(data: &[u8]) -> Result<Self, StorageError> {
        let header = SegmentHeader::from_bytes(data)?;
        if &header.magic != MAGIC_V2 && &header.magic != MAGIC_V3 {
            return Err(StorageError::Other("invalid segment magic".into()));
        }
        if header.version != 2 && header.version != 3 {
            return Err(StorageError::Other(format!(
                "unsupported segment version: {}",
                header.version
            )));
        }
        Ok(Self {
            vector_count: header.vector_count as usize,
            dimensions: header.dimensions as usize,
            flags: header.flags,
            norm_offset: header.norm_offset as usize,
            vector_offset: header.vector_offset as usize,
            id_offset: header.id_offset as usize,
            string_table_offset: header.string_table_offset as usize,
            attr_offset: header.attr_offset as usize,
            attr_len: header.attr_len as usize,
            text_offset: header.text_offset as usize,
            text_len: header.text_len as usize,
        })
    }

    #[inline]
    pub fn is_normalized(&self) -> bool {
        self.flags & flags::VECTORS_NORMALIZED != 0
    }
}

impl SegmentHeader {
    /// Parse header from bytes
    fn from_bytes(data: &[u8]) -> Result<Self, StorageError> {
        if data.len() < HEADER_SIZE_V2 {
            return Err(StorageError::Other("header too small".into()));
        }

        let mut cursor = std::io::Cursor::new(data);
        let mut magic = [0u8; 4];
        cursor
            .read_exact(&mut magic)
            .map_err(|e| StorageError::Other(e.to_string()))?;

        let version = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| StorageError::Other(e.to_string()))?;
        if (&magic == MAGIC_V2 && version != 2) || (&magic == MAGIC_V3 && version != 3) {
            return Err(StorageError::Other(format!(
                "unsupported segment version: {}",
                version
            )));
        }
        let vector_count = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| StorageError::Other(e.to_string()))?;
        let dimensions = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| StorageError::Other(e.to_string()))?;
        let flags = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| StorageError::Other(e.to_string()))?;
        let norm_offset = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| StorageError::Other(e.to_string()))?;
        let vector_offset = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| StorageError::Other(e.to_string()))?;
        let id_offset = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| StorageError::Other(e.to_string()))?;
        let string_table_offset = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| StorageError::Other(e.to_string()))?;
        let attr_offset = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| StorageError::Other(e.to_string()))?;
        let attr_len = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| StorageError::Other(e.to_string()))?;

        let (text_offset, text_len) = if &magic == MAGIC_V3 {
            if data.len() < HEADER_SIZE_V3 {
                return Err(StorageError::Other("header too small for v3".into()));
            }
            let text_offset = cursor
                .read_u64::<LittleEndian>()
                .map_err(|e| StorageError::Other(e.to_string()))?;
            let text_len = cursor
                .read_u64::<LittleEndian>()
                .map_err(|e| StorageError::Other(e.to_string()))?;
            (text_offset, text_len)
        } else {
            (0u64, 0u64)
        };

        Ok(Self {
            magic,
            version,
            vector_count,
            dimensions,
            flags,
            norm_offset,
            vector_offset,
            id_offset,
            string_table_offset,
            attr_offset,
            attr_len,
            text_offset,
            text_len,
        })
    }
}

/// View over the string ID lookup table
pub struct StringTableView<'a> {
    data: &'a [u8],
}

impl<'a> StringTableView<'a> {
    /// Look up string ID by index
    pub fn get(&self, index: usize) -> Option<&'a str> {
        let mut offset = 0usize;
        let mut current_index = 0usize;
        
        while offset + 4 <= self.data.len() {
            let len = u32::from_le_bytes([
                self.data[offset],
                self.data[offset + 1],
                self.data[offset + 2],
                self.data[offset + 3],
            ]) as usize;
            offset += 4;
            
            if offset + len > self.data.len() {
                return None;
            }
            
            if current_index == index {
                return std::str::from_utf8(&self.data[offset..offset + len]).ok();
            }
            
            offset += len;
            current_index += 1;
        }
        
        None
    }
}

/// Attribute data stored in segment
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SegmentAttrV2 {
    /// JSON-serialized attributes (empty if None)
    pub attributes_json: Vec<u8>,
}

impl<'a> SegmentView<'a> {
    /// Create a zero-copy view over segment data
    /// 
    /// # Safety
    /// The data must be a valid v2 segment file with proper alignment
    pub unsafe fn from_bytes(data: &'a [u8]) -> Result<Self, StorageError> {
        let header = SegmentHeader::from_bytes(data)?;
        
        if &header.magic != MAGIC_V2 && &header.magic != MAGIC_V3 {
            return Err(StorageError::Other("invalid segment magic".into()));
        }
        
        if header.version != 2 && header.version != 3 {
            return Err(StorageError::Other(format!(
                "unsupported segment version: {}",
                header.version
            )));
        }
        
        let n = header.vector_count as usize;
        let d = header.dimensions as usize;
        let norm_offset = header.norm_offset as usize;
        let vector_offset = header.vector_offset as usize;
        let id_offset = header.id_offset as usize;
        let string_table_offset = header.string_table_offset as usize;
        let attr_offset = header.attr_offset as usize;
        let attr_len = header.attr_len as usize;
        
        // Map norms
        let norms_bytes = n * mem::size_of::<f32>();
        if norm_offset + norms_bytes > data.len() {
            return Err(StorageError::Other("norms out of bounds".into()));
        }
        let norms = std::slice::from_raw_parts(
            data[norm_offset..].as_ptr() as *const f32,
            n,
        );
        
        // Map vectors
        let vectors_bytes = n * d * mem::size_of::<f32>();
        if vector_offset + vectors_bytes > data.len() {
            return Err(StorageError::Other("vectors out of bounds".into()));
        }
        let vectors = std::slice::from_raw_parts(
            data[vector_offset..].as_ptr() as *const f32,
            n * d,
        );
        
        // Map IDs
        let ids_bytes = n * mem::size_of::<u64>();
        if id_offset + ids_bytes > data.len() {
            return Err(StorageError::Other("ids out of bounds".into()));
        }
        let ids = std::slice::from_raw_parts(
            data[id_offset..].as_ptr() as *const u64,
            n,
        );
        
        // Map string table
        let string_table_len = attr_offset.saturating_sub(string_table_offset);
        let string_table = StringTableView {
            data: &data[string_table_offset..string_table_offset + string_table_len],
        };
        
        Ok(Self {
            data,
            vector_count: n,
            dimensions: d,
            flags: header.flags,
            attr_offset,
            attr_len,
            norms,
            vectors,
            ids,
            string_table,
        })
    }
    
    /// Number of vectors in segment
    #[inline]
    pub fn len(&self) -> usize {
        self.vector_count
    }
    
    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vector_count == 0
    }
    
    /// Vector dimensions
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    
    /// Check if vectors are pre-normalized
    #[inline]
    pub fn is_normalized(&self) -> bool {
        self.flags & flags::VECTORS_NORMALIZED != 0
    }
    
    /// Get vector by index (zero-copy)
    #[inline]
    pub fn vector(&self, index: usize) -> &'a [f32] {
        let d = self.dimensions;
        let start = index * d;
        &self.vectors[start..start + d]
    }
    
    /// Get precomputed norm for vector
    #[inline]
    pub fn norm(&self, index: usize) -> f32 {
        self.norms[index]
    }
    
    /// Get u64 ID for vector
    #[inline]
    pub fn id_hash(&self, index: usize) -> u64 {
        self.ids[index]
    }
    
    /// Get string ID for vector
    #[inline]
    pub fn id_string(&self, index: usize) -> Option<&'a str> {
        self.string_table.get(index)
    }
    
    /// Get attributes offset and length
    pub fn attr_region(&self) -> (&'a [u8], usize) {
        (&self.data[self.attr_offset..self.attr_offset + self.attr_len], self.attr_len)
    }
}

// Test-only methods for verifying segment correctness
#[cfg(test)]
impl<'a> SegmentView<'a> {
    /// Linear scan search (test only - production uses HNSW)
    pub fn linear_scan(
        &self,
        query: &[f32],
        query_norm: f32,
        top_k: usize,
        metric: DistanceMetric,
        allowed: Option<&RoaringBitmap>,
    ) -> Vec<(usize, f32)> {
        let mut results = Vec::with_capacity(self.len().min(top_k * 2));
        
        match allowed {
            Some(bitmap) => {
                for idx in bitmap.iter() {
                    let idx = idx as usize;
                    if idx >= self.len() {
                        continue;
                    }
                    let dist = self.distance_to(idx, query, query_norm, metric);
                    results.push((idx, dist));
                }
            }
            None => {
                for idx in 0..self.len() {
                    let dist = self.distance_to(idx, query, query_norm, metric);
                    results.push((idx, dist));
                }
            }
        }
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(top_k);
        results
    }
    
    /// Compute distance with precomputed norms
    #[inline]
    fn distance_to(&self, index: usize, query: &[f32], query_norm: f32, metric: DistanceMetric) -> f32 {
        let vec = self.vector(index);
        match metric {
            DistanceMetric::Cosine => {
                if self.is_normalized() && (query_norm - 1.0).abs() < 0.01 {
                    simd::cosine_distance_prenorm(query, vec)
                } else {
                    let vec_norm = self.norm(index);
                    simd::cosine_distance_with_norms(query, vec, query_norm, vec_norm)
                }
            }
            DistanceMetric::Euclidean => {
                simd::l2_squared_f32(query, vec)
            }
            DistanceMetric::DotProduct => {
                -simd::dot_f32(query, vec)
            }
        }
    }
}

/// Hash a string ID to u64
#[inline]
pub fn hash_id(id: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish()
}

/// Compute L2 norm of a vector (SIMD-accelerated)
#[inline]
pub fn compute_norm(vec: &[f32]) -> f32 {
    simd::l2_norm_f32(vec)
}

/// Align offset to boundary
#[inline]
fn align_to(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

/// Options for writing segment v2 files
#[derive(Debug, Clone, Default)]
pub struct WriteOptions {
    /// Pre-normalize vectors before writing
    /// When true, vectors are normalized to unit length and the VECTORS_NORMALIZED flag is set.
    /// This enables faster cosine distance computation: 1 - dot(a, b)
    pub normalize_vectors: bool,
}

/// Write a v2 segment file
pub fn write_segment_v2(docs: &[Document]) -> Result<Vec<u8>, StorageError> {
    write_segment_v2_with_options(docs, &WriteOptions::default())
}

/// Write a v2 segment file with options
pub fn write_segment_v2_with_options(docs: &[Document], opts: &WriteOptions) -> Result<Vec<u8>, StorageError> {
    if docs.is_empty() {
        return Err(StorageError::Other("no documents".into()));
    }
    
    let n = docs.len();
    let d = docs[0].vector.len();
    
    // Validate dimensions
    for (i, doc) in docs.iter().enumerate() {
        if doc.vector.len() != d {
            return Err(StorageError::Other(format!(
                "dimension mismatch at {}: {} != {}",
                i,
                doc.vector.len(),
                d
            )));
        }
    }
    
    // Calculate offsets
    let norm_offset = align_to(HEADER_SIZE_V2, ALIGNMENT);
    let norms_size = n * mem::size_of::<f32>();
    
    let vector_offset = align_to(norm_offset + norms_size, ALIGNMENT);
    let vectors_size = n * d * mem::size_of::<f32>();
    
    let id_offset = align_to(vector_offset + vectors_size, 8);
    let ids_size = n * mem::size_of::<u64>();
    
    let string_table_offset = id_offset + ids_size;
    
    // Build string table
    let mut string_table = Vec::new();
    for doc in docs {
        let id_bytes = doc.id.as_bytes();
        string_table.write_u32::<LittleEndian>(id_bytes.len() as u32)
            .map_err(|e| StorageError::Other(e.to_string()))?;
        string_table.extend_from_slice(id_bytes);
    }
    
    // Align attr_offset to 4 bytes for rkyv
    let attr_offset = align_to(string_table_offset + string_table.len(), 4);
    
    // Build attributes
    let attrs: Vec<SegmentAttrV2> = docs
        .iter()
        .map(|doc| {
            let json = match &doc.attributes {
                Some(attrs) => serde_json::to_vec(attrs).unwrap_or_default(),
                None => Vec::new(),
            };
            SegmentAttrV2 { attributes_json: json }
        })
        .collect();
    
    let attr_bytes = rkyv::to_bytes::<_, 256>(&attrs)
        .map_err(|e| StorageError::Other(e.to_string()))?;
    let attr_len = attr_bytes.len();
    
    // Allocate buffer (attr_offset already aligned)
    let total_size = attr_offset + attr_len;
    let mut buf = vec![0u8; total_size];
    
    // Set flags
    let segment_flags = if opts.normalize_vectors {
        flags::VECTORS_NORMALIZED
    } else {
        0
    };
    
    // Write header
    buf[0..4].copy_from_slice(MAGIC_V2);
    (&mut buf[4..8]).write_u32::<LittleEndian>(2).unwrap();
    (&mut buf[8..12]).write_u32::<LittleEndian>(n as u32).unwrap();
    (&mut buf[12..16]).write_u32::<LittleEndian>(d as u32).unwrap();
    (&mut buf[16..20]).write_u32::<LittleEndian>(segment_flags).unwrap();
    (&mut buf[20..28]).write_u64::<LittleEndian>(norm_offset as u64).unwrap();
    (&mut buf[28..36]).write_u64::<LittleEndian>(vector_offset as u64).unwrap();
    (&mut buf[36..44]).write_u64::<LittleEndian>(id_offset as u64).unwrap();
    (&mut buf[44..52]).write_u64::<LittleEndian>(string_table_offset as u64).unwrap();
    (&mut buf[52..60]).write_u64::<LittleEndian>(attr_offset as u64).unwrap();
    (&mut buf[60..68]).write_u64::<LittleEndian>(attr_len as u64).unwrap();
    
    // Write norms and optionally normalized vectors
    let mut offset = norm_offset;
    
    if opts.normalize_vectors {
        // Compute norms first, then write normalized vectors
        let mut normalized_vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
        for doc in docs {
            let norm = simd::l2_norm_f32(&doc.vector);
            (&mut buf[offset..offset + 4]).write_f32::<LittleEndian>(1.0).unwrap(); // norm is 1 after normalization
            offset += 4;
            
            if norm == 0.0 {
                normalized_vecs.push(doc.vector.clone());
            } else {
                normalized_vecs.push(simd::normalized_f32(&doc.vector));
            }
        }
        
        // Write normalized vectors
        offset = vector_offset;
        for vec in &normalized_vecs {
            for &v in vec {
                (&mut buf[offset..offset + 4]).write_f32::<LittleEndian>(v).unwrap();
                offset += 4;
            }
        }
    } else {
        // Write original norms
        for doc in docs {
            let norm = simd::l2_norm_f32(&doc.vector);
            (&mut buf[offset..offset + 4]).write_f32::<LittleEndian>(norm).unwrap();
            offset += 4;
        }
        
        // Write original vectors
        offset = vector_offset;
        for doc in docs {
            for &v in &doc.vector {
                (&mut buf[offset..offset + 4]).write_f32::<LittleEndian>(v).unwrap();
                offset += 4;
            }
        }
    }
    
    // Write IDs
    offset = id_offset;
    for doc in docs {
        let id_hash = hash_id(&doc.id);
        (&mut buf[offset..offset + 8]).write_u64::<LittleEndian>(id_hash).unwrap();
        offset += 8;
    }
    
    // Write string table
    buf[string_table_offset..string_table_offset + string_table.len()]
        .copy_from_slice(&string_table);
    
    // Write attributes
    buf[attr_offset..attr_offset + attr_len].copy_from_slice(&attr_bytes);
    
    Ok(buf)
}

/// Write a v3 segment file (includes text table)
pub fn write_segment_v3(docs: &[Document]) -> Result<Vec<u8>, StorageError> {
    write_segment_v3_with_options(docs, &WriteOptions::default())
}

/// Write a v3 segment file with options
pub fn write_segment_v3_with_options(docs: &[Document], opts: &WriteOptions) -> Result<Vec<u8>, StorageError> {
    if docs.is_empty() {
        return Err(StorageError::Other("no documents".into()));
    }

    let n = docs.len();
    let d = docs[0].vector.len();

    // Validate dimensions
    for (i, doc) in docs.iter().enumerate() {
        if doc.vector.len() != d {
            return Err(StorageError::Other(format!(
                "dimension mismatch at {}: {} != {}",
                i,
                doc.vector.len(),
                d
            )));
        }
    }

    // Calculate offsets
    let norm_offset = align_to(HEADER_SIZE_V3, ALIGNMENT);
    let norms_size = n * mem::size_of::<f32>();

    let vector_offset = align_to(norm_offset + norms_size, ALIGNMENT);
    let vectors_size = n * d * mem::size_of::<f32>();

    let id_offset = align_to(vector_offset + vectors_size, 8);
    let ids_size = n * mem::size_of::<u64>();

    let string_table_offset = id_offset + ids_size;

    // Build string table
    let mut string_table = Vec::new();
    for doc in docs {
        let id_bytes = doc.id.as_bytes();
        string_table
            .write_u32::<LittleEndian>(id_bytes.len() as u32)
            .map_err(|e| StorageError::Other(e.to_string()))?;
        string_table.extend_from_slice(id_bytes);
    }

    // Align attr_offset to 4 bytes for rkyv
    let attr_offset = align_to(string_table_offset + string_table.len(), 4);

    // Build attributes
    let attrs: Vec<SegmentAttrV2> = docs
        .iter()
        .map(|doc| {
            let json = match &doc.attributes {
                Some(attrs) => serde_json::to_vec(attrs).unwrap_or_default(),
                None => Vec::new(),
            };
            SegmentAttrV2 { attributes_json: json }
        })
        .collect();

    let attr_bytes = rkyv::to_bytes::<_, 256>(&attrs)
        .map_err(|e| StorageError::Other(e.to_string()))?;
    let attr_len = attr_bytes.len();

    // Build text table (length-prefixed per doc)
    let mut text_table = Vec::new();
    for doc in docs {
        let text = doc.text.as_deref().unwrap_or("");
        let bytes = text.as_bytes();
        text_table
            .write_u32::<LittleEndian>(bytes.len() as u32)
            .map_err(|e| StorageError::Other(e.to_string()))?;
        text_table.extend_from_slice(bytes);
    }

    let text_offset = align_to(attr_offset + attr_len, 4);
    let text_len = text_table.len();

    // Allocate buffer
    let total_size = text_offset + text_len;
    let mut buf = vec![0u8; total_size];

    // Set flags
    let segment_flags = if opts.normalize_vectors {
        flags::VECTORS_NORMALIZED
    } else {
        0
    };

    // Write header
    buf[0..4].copy_from_slice(MAGIC_V3);
    (&mut buf[4..8]).write_u32::<LittleEndian>(3).unwrap();
    (&mut buf[8..12]).write_u32::<LittleEndian>(n as u32).unwrap();
    (&mut buf[12..16]).write_u32::<LittleEndian>(d as u32).unwrap();
    (&mut buf[16..20]).write_u32::<LittleEndian>(segment_flags).unwrap();
    (&mut buf[20..28]).write_u64::<LittleEndian>(norm_offset as u64).unwrap();
    (&mut buf[28..36]).write_u64::<LittleEndian>(vector_offset as u64).unwrap();
    (&mut buf[36..44]).write_u64::<LittleEndian>(id_offset as u64).unwrap();
    (&mut buf[44..52]).write_u64::<LittleEndian>(string_table_offset as u64).unwrap();
    (&mut buf[52..60]).write_u64::<LittleEndian>(attr_offset as u64).unwrap();
    (&mut buf[60..68]).write_u64::<LittleEndian>(attr_len as u64).unwrap();
    (&mut buf[68..76]).write_u64::<LittleEndian>(text_offset as u64).unwrap();
    (&mut buf[76..84]).write_u64::<LittleEndian>(text_len as u64).unwrap();

    // Write norms and optionally normalized vectors
    let mut offset = norm_offset;

    if opts.normalize_vectors {
        // Compute norms first, then write normalized vectors
        let mut normalized_vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
        for doc in docs {
            let norm = simd::l2_norm_f32(&doc.vector);
            (&mut buf[offset..offset + 4]).write_f32::<LittleEndian>(1.0).unwrap();
            offset += 4;

            if norm == 0.0 {
                normalized_vecs.push(doc.vector.clone());
            } else {
                normalized_vecs.push(simd::normalized_f32(&doc.vector));
            }
        }

        // Write normalized vectors
        offset = vector_offset;
        for vec in &normalized_vecs {
            for &v in vec {
                (&mut buf[offset..offset + 4]).write_f32::<LittleEndian>(v).unwrap();
                offset += 4;
            }
        }
    } else {
        // Write original norms
        for doc in docs {
            let norm = simd::l2_norm_f32(&doc.vector);
            (&mut buf[offset..offset + 4]).write_f32::<LittleEndian>(norm).unwrap();
            offset += 4;
        }

        // Write original vectors
        offset = vector_offset;
        for doc in docs {
            for &v in &doc.vector {
                (&mut buf[offset..offset + 4]).write_f32::<LittleEndian>(v).unwrap();
                offset += 4;
            }
        }
    }

    // Write IDs
    offset = id_offset;
    for doc in docs {
        let id_hash = hash_id(&doc.id);
        (&mut buf[offset..offset + 8]).write_u64::<LittleEndian>(id_hash).unwrap();
        offset += 8;
    }

    // Write string table
    buf[string_table_offset..string_table_offset + string_table.len()]
        .copy_from_slice(&string_table);

    // Write attributes
    buf[attr_offset..attr_offset + attr_len].copy_from_slice(&attr_bytes);

    // Write text table
    buf[text_offset..text_offset + text_len].copy_from_slice(&text_table);

    Ok(buf)
}

/// Check if data is v2 format
pub fn is_v2_format(data: &[u8]) -> bool {
    data.len() >= 4 && &data[0..4] == MAGIC_V2
}

/// Check if data is v3 format
pub fn is_v3_format(data: &[u8]) -> bool {
    data.len() >= 4 && &data[0..4] == MAGIC_V3
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_roundtrip() {
        let docs = vec![
            Document {
                id: "doc1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                text: None,
                attributes: None,
            },
            Document {
                id: "doc2".to_string(),
                vector: vec![0.0, 1.0, 0.0],
                text: None,
                attributes: None,
            },
        ];
        
        let data = write_segment_v2(&docs).unwrap();
        assert!(is_v2_format(&data));
        
        let view = unsafe { SegmentView::from_bytes(&data).unwrap() };
        assert_eq!(view.len(), 2);
        assert_eq!(view.dimensions(), 3);
        assert!(!view.is_normalized());
        
        assert_eq!(view.id_string(0), Some("doc1"));
        assert_eq!(view.id_string(1), Some("doc2"));
        
        let v0 = view.vector(0);
        assert_eq!(v0, &[1.0, 0.0, 0.0]);
        
        // Check norm
        assert!((view.norm(0) - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_roundtrip_normalized() {
        let docs = vec![
            Document {
                id: "doc1".to_string(),
                vector: vec![3.0, 4.0, 0.0],
                text: None,
                attributes: None,
            },
            Document {
                id: "doc2".to_string(),
                vector: vec![0.0, 5.0, 12.0],
                text: None,
                attributes: None,
            },
        ];
        
        let opts = WriteOptions { normalize_vectors: true };
        let data = write_segment_v2_with_options(&docs, &opts).unwrap();
        
        let view = unsafe { SegmentView::from_bytes(&data).unwrap() };
        assert!(view.is_normalized());
        
        // Check norms are 1.0
        assert!((view.norm(0) - 1.0).abs() < 0.001);
        assert!((view.norm(1) - 1.0).abs() < 0.001);
        
        // Check vectors are normalized
        let v0 = view.vector(0);
        let v0_norm = simd::l2_norm_f32(v0);
        assert!((v0_norm - 1.0).abs() < 0.001, "vector 0 norm: {}", v0_norm);
        
        // Original: [3, 4, 0], norm = 5, normalized = [0.6, 0.8, 0]
        assert!((v0[0] - 0.6).abs() < 0.001);
        assert!((v0[1] - 0.8).abs() < 0.001);
    }
    
    #[test]
    fn test_search() {
        let docs = vec![
            Document {
                id: "a".to_string(),
                vector: vec![1.0, 0.0],
                text: None,
                attributes: None,
            },
            Document {
                id: "b".to_string(),
                vector: vec![0.0, 1.0],
                text: None,
                attributes: None,
            },
            Document {
                id: "c".to_string(),
                vector: vec![0.7, 0.7],
                text: None,
                attributes: None,
            },
        ];
        
        let data = write_segment_v2(&docs).unwrap();
        let view = unsafe { SegmentView::from_bytes(&data).unwrap() };
        
        let query = vec![1.0, 0.0];
        let query_norm = compute_norm(&query);
        
        let results = view.linear_scan(&query, query_norm, 2, DistanceMetric::Cosine, None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // "a" is closest
    }
    
    #[test]
    fn test_search_normalized() {
        let docs = vec![
            Document {
                id: "a".to_string(),
                vector: vec![1.0, 0.0],
                text: None,
                attributes: None,
            },
            Document {
                id: "b".to_string(),
                vector: vec![0.0, 1.0],
                text: None,
                attributes: None,
            },
            Document {
                id: "c".to_string(),
                vector: vec![0.7, 0.7],
                text: None,
                attributes: None,
            },
        ];
        
        let opts = WriteOptions { normalize_vectors: true };
        let data = write_segment_v2_with_options(&docs, &opts).unwrap();
        let view = unsafe { SegmentView::from_bytes(&data).unwrap() };
        
        // Query should also be normalized for fast path
        let query = simd::normalized_f32(&[1.0, 0.0]);
        let query_norm = 1.0;
        
        let results = view.linear_scan(&query, query_norm, 2, DistanceMetric::Cosine, None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // "a" is closest
    }
}
