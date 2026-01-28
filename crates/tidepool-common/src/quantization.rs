//! Quantization codecs and distance helpers.
//!
//! Provides f16 and SQ8 (per-dimension asymmetric) quantization with a
//! simple binary sidecar format for segment data.

use std::io::{Cursor, Read};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use half::f16;

use crate::vector::DistanceMetric;

const MAGIC: &[u8; 4] = b"TPQ1";
const VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationKind {
    None,
    F16,
    SQ8,
}

impl QuantizationKind {
    pub fn parse(raw: Option<&str>) -> Self {
        match raw.unwrap_or("").to_lowercase().as_str() {
            "f16" => Self::F16,
            "sq8" => Self::SQ8,
            _ => Self::None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::F16 => "f16",
            Self::SQ8 => "sq8",
        }
    }

    fn to_u32(self) -> u32 {
        match self {
            Self::None => 0,
            Self::F16 => 1,
            Self::SQ8 => 2,
        }
    }

    fn from_u32(value: u32) -> Result<Self, String> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::F16),
            2 => Ok(Self::SQ8),
            _ => Err(format!("unsupported quantization kind: {}", value)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedVectors {
    pub kind: QuantizationKind,
    pub dimensions: usize,
    pub vector_count: usize,
    /// Raw vector data. For f16: little-endian u16 per element.
    /// For SQ8: u8 code per element.
    pub data: Vec<u8>,
    /// Per-dimension scales for SQ8 (len = dimensions).
    pub scales: Vec<f32>,
    /// Per-dimension mins for SQ8 (len = dimensions).
    pub mins: Vec<f32>,
}

impl QuantizedVectors {
    pub fn vector_offset(&self, index: usize) -> Option<usize> {
        if index >= self.vector_count {
            return None;
        }
        let stride = match self.kind {
            QuantizationKind::F16 => self.dimensions * 2,
            QuantizationKind::SQ8 => self.dimensions,
            QuantizationKind::None => 0,
        };
        Some(index * stride)
    }

    pub fn vector_bytes(&self, index: usize) -> Option<&[u8]> {
        let start = self.vector_offset(index)?;
        let len = match self.kind {
            QuantizationKind::F16 => self.dimensions * 2,
            QuantizationKind::SQ8 => self.dimensions,
            QuantizationKind::None => 0,
        };
        if start + len > self.data.len() {
            return None;
        }
        Some(&self.data[start..start + len])
    }
}

pub fn quantize(vectors: &[&[f32]], kind: QuantizationKind) -> Result<QuantizedVectors, String> {
    match kind {
        QuantizationKind::None => Err("quantization kind is none".into()),
        QuantizationKind::F16 => quantize_f16(vectors),
        QuantizationKind::SQ8 => quantize_sq8(vectors),
    }
}

pub fn quantize_f16(vectors: &[&[f32]]) -> Result<QuantizedVectors, String> {
    if vectors.is_empty() {
        return Err("no vectors".into());
    }
    let dims = vectors[0].len();
    if dims == 0 {
        return Err("invalid dimensions".into());
    }
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != dims {
            return Err(format!("dimension mismatch at {}: {} != {}", i, v.len(), dims));
        }
    }

    let mut data = Vec::with_capacity(vectors.len() * dims * 2);
    for v in vectors {
        for &x in *v {
            let bits = f16::from_f32(x).to_bits();
            data.extend_from_slice(&bits.to_le_bytes());
        }
    }

    Ok(QuantizedVectors {
        kind: QuantizationKind::F16,
        dimensions: dims,
        vector_count: vectors.len(),
        data,
        scales: Vec::new(),
        mins: Vec::new(),
    })
}

pub fn quantize_sq8(vectors: &[&[f32]]) -> Result<QuantizedVectors, String> {
    if vectors.is_empty() {
        return Err("no vectors".into());
    }
    let dims = vectors[0].len();
    if dims == 0 {
        return Err("invalid dimensions".into());
    }
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != dims {
            return Err(format!("dimension mismatch at {}: {} != {}", i, v.len(), dims));
        }
    }

    let mut mins = vec![f32::INFINITY; dims];
    let mut maxs = vec![f32::NEG_INFINITY; dims];
    for v in vectors {
        for (i, &x) in (*v).iter().enumerate() {
            if x.is_nan() {
                continue;
            }
            if x < mins[i] {
                mins[i] = x;
            }
            if x > maxs[i] {
                maxs[i] = x;
            }
        }
    }

    let mut scales = vec![0.0f32; dims];
    for i in 0..dims {
        let min = mins[i];
        let max = maxs[i];
        if !min.is_finite() || !max.is_finite() {
            mins[i] = 0.0;
            scales[i] = 1.0;
        } else if max <= min {
            scales[i] = 1.0;
        } else {
            scales[i] = (max - min) / 255.0;
        }
    }

    let mut data = Vec::with_capacity(vectors.len() * dims);
    for v in vectors {
        for (i, &x) in (*v).iter().enumerate() {
            let min = mins[i];
            let scale = scales[i];
            let code = if scale == 0.0 {
                0u8
            } else {
                let normalized = (x - min) / scale;
                normalized.round().clamp(0.0, 255.0) as u8
            };
            data.push(code);
        }
    }

    Ok(QuantizedVectors {
        kind: QuantizationKind::SQ8,
        dimensions: dims,
        vector_count: vectors.len(),
        data,
        scales,
        mins,
    })
}

pub fn marshal_binary(q: &QuantizedVectors) -> Result<Vec<u8>, String> {
    if q.kind == QuantizationKind::None {
        return Err("cannot marshal quantization kind none".into());
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.write_u32::<LittleEndian>(VERSION).map_err(|e| e.to_string())?;
    buf.write_u32::<LittleEndian>(q.kind.to_u32())
        .map_err(|e| e.to_string())?;
    buf.write_u32::<LittleEndian>(q.vector_count as u32)
        .map_err(|e| e.to_string())?;
    buf.write_u32::<LittleEndian>(q.dimensions as u32)
        .map_err(|e| e.to_string())?;

    if q.kind == QuantizationKind::SQ8 {
        if q.scales.len() != q.dimensions || q.mins.len() != q.dimensions {
            return Err("invalid SQ8 calibration data".into());
        }
        for s in &q.scales {
            buf.write_f32::<LittleEndian>(*s).map_err(|e| e.to_string())?;
        }
        for m in &q.mins {
            buf.write_f32::<LittleEndian>(*m).map_err(|e| e.to_string())?;
        }
    }

    buf.extend_from_slice(&q.data);
    Ok(buf)
}

pub fn load_binary(data: &[u8]) -> Result<QuantizedVectors, String> {
    let mut cursor = Cursor::new(data);
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC {
        return Err("invalid quantization format".into());
    }
    let version = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
    if version != VERSION {
        return Err(format!("unsupported quantization version: {}", version));
    }

    let kind = QuantizationKind::from_u32(cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?)?;
    if kind == QuantizationKind::None {
        return Err("invalid quantization kind none".into());
    }
    let vector_count = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
    let dimensions = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
    if vector_count == 0 || dimensions == 0 {
        return Err("invalid quantization header".into());
    }

    let mut scales = Vec::new();
    let mut mins = Vec::new();
    if kind == QuantizationKind::SQ8 {
        scales.resize(dimensions, 0.0);
        mins.resize(dimensions, 0.0);
        for s in &mut scales {
            *s = cursor.read_f32::<LittleEndian>().map_err(|e| e.to_string())?;
        }
        for m in &mut mins {
            *m = cursor.read_f32::<LittleEndian>().map_err(|e| e.to_string())?;
        }
    }

    let mut payload = Vec::new();
    cursor.read_to_end(&mut payload).map_err(|e| e.to_string())?;

    let expected = match kind {
        QuantizationKind::F16 => vector_count * dimensions * 2,
        QuantizationKind::SQ8 => vector_count * dimensions,
        QuantizationKind::None => 0,
    };
    if payload.len() != expected {
        return Err(format!("quantization payload mismatch: got {} want {}", payload.len(), expected));
    }

    Ok(QuantizedVectors {
        kind,
        dimensions,
        vector_count,
        data: payload,
        scales,
        mins,
    })
}

pub struct Sq8Query {
    scaled: Vec<f32>,
    min_dot: f32,
    query_norm: f32,
    query: Vec<f32>,
}

impl Sq8Query {
    pub fn new(query: &[f32], scales: &[f32], mins: &[f32], query_norm: f32) -> Self {
        let mut scaled = Vec::with_capacity(query.len());
        let mut min_dot = 0.0f32;
        for i in 0..query.len() {
            let q = query[i];
            let min = mins[i];
            let scale = scales[i];
            min_dot += q * min;
            scaled.push(q * scale);
        }
        Self { scaled, min_dot, query_norm, query: query.to_vec() }
    }
}

#[inline]
pub fn sq8_distance(
    query: &Sq8Query,
    codes: &[u8],
    scales: &[f32],
    mins: &[f32],
    metric: DistanceMetric,
    vector_norm: f32,
) -> f32 {
    match metric {
        DistanceMetric::DotProduct => {
            let mut dot = query.min_dot;
            for (i, &code) in codes.iter().enumerate() {
                dot += query.scaled[i] * (code as f32);
            }
            -dot
        }
        DistanceMetric::Cosine => {
            if query.query_norm == 0.0 || vector_norm == 0.0 {
                return 2.0;
            }
            let mut dot = query.min_dot;
            for (i, &code) in codes.iter().enumerate() {
                dot += query.scaled[i] * (code as f32);
            }
            1.0 - dot / (query.query_norm * vector_norm)
        }
        DistanceMetric::Euclidean => {
            let mut sum = 0.0f32;
            for i in 0..codes.len() {
                let v = mins[i] + (codes[i] as f32) * scales[i];
                let d = query.query[i] - v;
                sum += d * d;
            }
            sum
        }
    }
}

#[inline]
pub fn f16_distance(
    query: &[f32],
    bytes: &[u8],
    metric: DistanceMetric,
    query_norm: f32,
    vector_norm: f32,
) -> f32 {
    let dims = query.len();
    let mut dot = 0.0f32;
    let mut l2 = 0.0f32;
    for i in 0..dims {
        let base = i * 2;
        let bits = u16::from_le_bytes([bytes[base], bytes[base + 1]]);
        let v = f16::from_bits(bits).to_f32();
        match metric {
            DistanceMetric::Euclidean => {
                let d = query[i] - v;
                l2 += d * d;
            }
            _ => {
                dot += query[i] * v;
            }
        }
    }

    match metric {
        DistanceMetric::DotProduct => -dot,
        DistanceMetric::Cosine => {
            if query_norm == 0.0 || vector_norm == 0.0 {
                2.0
            } else {
                1.0 - dot / (query_norm * vector_norm)
            }
        }
        DistanceMetric::Euclidean => l2,
    }
}
