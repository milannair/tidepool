use std::io::{Cursor, Read};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;

use crate::simd;
use crate::vector::{distance_with_norms, DistanceMetric};

const MAGIC: &[u8; 4] = b"TPIV";
const VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct IVFIndex {
    pub k: usize,
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub nprobe_default: usize,
    /// Flattened centroid matrix (k * dimensions)
    pub centroids: Vec<f32>,
    /// Precomputed centroid norms (for cosine)
    centroid_norms: Vec<f32>,
    /// Per-cluster radius (metric distance; for Euclidean this is L2, not squared)
    pub radii: Vec<f32>,
    /// Posting lists per centroid
    pub posting_lists: Vec<Vec<u32>>,
}

impl IVFIndex {
    pub fn compute_k(vector_count: usize, k_factor: f32, min_k: usize, max_k: usize) -> usize {
        if vector_count == 0 {
            return 0;
        }
        let mut k = ((vector_count as f64).sqrt() * k_factor as f64).round() as usize;
        if k == 0 {
            k = 1;
        }
        let min_k = min_k.max(1).min(vector_count);
        let max_k = max_k.max(min_k).min(vector_count);
        k = k.clamp(min_k, max_k);
        k
    }

    pub fn centroid(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimensions;
        &self.centroids[start..start + self.dimensions]
    }

    pub fn centroid_norm(&self, idx: usize) -> f32 {
        self.centroid_norms[idx]
    }

    pub fn build(
        vectors: &[&[f32]],
        metric: DistanceMetric,
        k: usize,
        nprobe_default: usize,
        max_iters: usize,
        seed: u64,
    ) -> Result<Self, String> {
        if vectors.is_empty() {
            return Err("no vectors".into());
        }
        let dims = vectors[0].len();
        if dims == 0 {
            return Err("invalid vector dimensions".into());
        }
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dims {
                return Err(format!("dimension mismatch at {}: {} != {}", i, v.len(), dims));
            }
        }

        let k = k.clamp(1, vectors.len());
        let nprobe_default = nprobe_default.clamp(1, k);
        let max_iters = max_iters.max(1);

        let mut rng = StdRng::seed_from_u64(seed);

        let mut indices: Vec<usize> = (0..vectors.len()).collect();
        indices.shuffle(&mut rng);

        let mut centroids = vec![0.0f32; k * dims];
        for c in 0..k {
            let idx = indices[c % indices.len()];
            let start = c * dims;
            centroids[start..start + dims].copy_from_slice(vectors[idx]);
        }

        let mut centroid_norms = vec![0.0f32; k];
        if metric == DistanceMetric::Cosine {
            for c in 0..k {
                let start = c * dims;
                let norm = simd::l2_norm_f32(&centroids[start..start + dims]);
                if norm > 0.0 {
                    for d in 0..dims {
                        centroids[start + d] /= norm;
                    }
                    centroid_norms[c] = 1.0;
                } else {
                    centroid_norms[c] = 0.0;
                }
            }
        }

        let mut vec_norms = vec![0.0f32; vectors.len()];
        if metric == DistanceMetric::Cosine {
            for (i, v) in vectors.iter().enumerate() {
                vec_norms[i] = simd::l2_norm_f32(v);
            }
        }

        let mut assignments = vec![0usize; vectors.len()];

        for _ in 0..max_iters {
            let mut changed = false;

            for (i, v) in vectors.iter().enumerate() {
                let mut best = 0usize;
                let mut best_dist = f32::INFINITY;
                for c in 0..k {
                    let c_start = c * dims;
                    let dist = distance_with_norms(
                        v,
                        &centroids[c_start..c_start + dims],
                        vec_norms[i],
                        centroid_norms[c],
                        metric,
                    );
                    if dist < best_dist {
                        best_dist = dist;
                        best = c;
                    }
                }
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            let mut counts = vec![0usize; k];
            let mut sums = vec![0.0f32; k * dims];
            for (i, v) in vectors.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                let base = c * dims;
                for d in 0..dims {
                    sums[base + d] += v[d];
                }
            }

            for c in 0..k {
                let base = c * dims;
                if counts[c] == 0 {
                    let idx = rng.gen_range(0..vectors.len());
                    centroids[base..base + dims].copy_from_slice(vectors[idx]);
                } else {
                    let inv = 1.0 / counts[c] as f32;
                    for d in 0..dims {
                        centroids[base + d] = sums[base + d] * inv;
                    }
                }

                if metric == DistanceMetric::Cosine {
                    let norm = simd::l2_norm_f32(&centroids[base..base + dims]);
                    if norm > 0.0 {
                        for d in 0..dims {
                            centroids[base + d] /= norm;
                        }
                        centroid_norms[c] = 1.0;
                    } else {
                        centroid_norms[c] = 0.0;
                    }
                }
            }
        }

        if metric != DistanceMetric::Cosine {
            centroid_norms.clear();
            centroid_norms.resize(k, 0.0);
        }

        let mut posting_lists: Vec<Vec<u32>> = vec![Vec::new(); k];
        for (i, &c) in assignments.iter().enumerate() {
            posting_lists[c].push(i as u32);
        }

        let mut radii = vec![0.0f32; k];
        for (i, v) in vectors.iter().enumerate() {
            let c = assignments[i];
            let base = c * dims;
            let dist = distance_with_norms(
                v,
                &centroids[base..base + dims],
                vec_norms[i],
                centroid_norms[c],
                metric,
            );
            let dist = if metric == DistanceMetric::Euclidean {
                dist.sqrt()
            } else {
                dist
            };
            if dist > radii[c] {
                radii[c] = dist;
            }
        }

        Ok(Self {
            k,
            dimensions: dims,
            metric,
            nprobe_default,
            centroids,
            centroid_norms,
            radii,
            posting_lists,
        })
    }

    pub fn marshal_binary(&self) -> Result<Vec<u8>, String> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.write_u32::<LittleEndian>(VERSION).map_err(|e| e.to_string())?;
        buf.write_u32::<LittleEndian>(self.k as u32)
            .map_err(|e| e.to_string())?;
        buf.write_u32::<LittleEndian>(self.dimensions as u32)
            .map_err(|e| e.to_string())?;

        let metric = self.metric.as_str().as_bytes();
        buf.write_u32::<LittleEndian>(metric.len() as u32)
            .map_err(|e| e.to_string())?;
        buf.extend_from_slice(metric);
        buf.write_u32::<LittleEndian>(self.nprobe_default as u32)
            .map_err(|e| e.to_string())?;

        for r in &self.radii {
            buf.write_f32::<LittleEndian>(*r).map_err(|e| e.to_string())?;
        }
        for v in &self.centroids {
            buf.write_f32::<LittleEndian>(*v).map_err(|e| e.to_string())?;
        }
        for list in &self.posting_lists {
            buf.write_u32::<LittleEndian>(list.len() as u32)
                .map_err(|e| e.to_string())?;
            for &id in list {
                buf.write_u32::<LittleEndian>(id).map_err(|e| e.to_string())?;
            }
        }

        Ok(buf)
    }

    pub fn load_binary(data: &[u8]) -> Result<Self, String> {
        let mut cursor = Cursor::new(data);
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != MAGIC {
            return Err("invalid IVF format".into());
        }
        let version = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        if version != VERSION {
            return Err(format!("unsupported IVF version: {}", version));
        }
        let k = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
        let dimensions = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
        if k == 0 || dimensions == 0 {
            return Err("invalid IVF header".into());
        }

        let metric_len = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
        if metric_len == 0 {
            return Err("missing IVF metric".into());
        }
        let mut metric_bytes = vec![0u8; metric_len];
        cursor.read_exact(&mut metric_bytes).map_err(|e| e.to_string())?;
        let metric_str = String::from_utf8_lossy(&metric_bytes).to_string();
        let metric = DistanceMetric::parse(Some(metric_str.as_str()));

        let nprobe_default = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;

        let mut radii = vec![0.0f32; k];
        for r in &mut radii {
            *r = cursor.read_f32::<LittleEndian>().map_err(|e| e.to_string())?;
        }

        let mut centroids = vec![0.0f32; k * dimensions];
        for v in &mut centroids {
            *v = cursor.read_f32::<LittleEndian>().map_err(|e| e.to_string())?;
        }

        let mut posting_lists: Vec<Vec<u32>> = Vec::with_capacity(k);
        for _ in 0..k {
            let len = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                list.push(cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?);
            }
            posting_lists.push(list);
        }

        let mut centroid_norms = vec![0.0f32; k];
        if metric == DistanceMetric::Cosine {
            for c in 0..k {
                let start = c * dimensions;
                centroid_norms[c] = simd::l2_norm_f32(&centroids[start..start + dimensions]);
            }
        }

        Ok(Self {
            k,
            dimensions,
            metric,
            nprobe_default: nprobe_default.clamp(1, k),
            centroids,
            centroid_norms,
            radii,
            posting_lists,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ivf_roundtrip() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let ivf = IVFIndex::build(&refs, DistanceMetric::Cosine, 2, 2, 5, 123)
            .expect("build ivf");
        let data = ivf.marshal_binary().expect("marshal ivf");
        let loaded = IVFIndex::load_binary(&data).expect("load ivf");

        assert_eq!(ivf.k, loaded.k);
        assert_eq!(ivf.dimensions, loaded.dimensions);
        assert_eq!(ivf.metric, loaded.metric);
        assert_eq!(ivf.nprobe_default, loaded.nprobe_default);
        assert_eq!(ivf.radii.len(), loaded.radii.len());
        assert_eq!(ivf.centroids.len(), loaded.centroids.len());
        assert_eq!(ivf.posting_lists.len(), loaded.posting_lists.len());
    }
}
