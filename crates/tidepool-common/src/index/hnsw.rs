use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{Cursor, Read};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use roaring::RoaringBitmap;

use crate::vector::{distance, DistanceMetric};

pub const DEFAULT_M: usize = 16;
pub const DEFAULT_EF_CONSTRUCTION: usize = 200;
pub const DEFAULT_EF_SEARCH: usize = 100;

#[derive(Debug, Clone)]
pub struct HnswIndex {
    pub nodes: Vec<HnswNode>,
    pub entry_point: isize,
    pub max_level: isize,
    pub m: usize,
    pub ml: f64,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub metric: DistanceMetric,
    rng: StdRng,
}

#[derive(Debug, Clone)]
pub struct HnswNode {
    pub id: usize,
    pub vector: Vec<f32>,
    pub connections: Vec<Vec<usize>>, // per level
    pub level: usize,
}

#[derive(Debug, Clone)]
pub struct ResultItem {
    pub id: usize,
    pub dist: f32,
}

impl HnswIndex {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize, metric: DistanceMetric) -> Self {
        let mut m = if m == 0 { DEFAULT_M } else { m };
        if m < 2 {
            m = 2;
        }
        let mut ef_construction = if ef_construction == 0 {
            DEFAULT_EF_CONSTRUCTION
        } else {
            ef_construction
        };
        if ef_construction < m {
            ef_construction = m;
        }
        let ef_search = if ef_search == 0 { DEFAULT_EF_SEARCH } else { ef_search };
        let ml = 1.0 / (m as f64).ln();
        Self {
            nodes: Vec::new(),
            entry_point: -1,
            max_level: -1,
            m,
            ml,
            ef_construction,
            ef_search,
            metric,
            rng: StdRng::from_entropy(),
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    pub fn insert(&mut self, id: usize, vector: Vec<f32>) {
        let level = self.random_level();
        let node = HnswNode {
            id,
            vector,
            connections: vec![Vec::new(); level + 1],
            level,
        };

        if id == self.nodes.len() {
            self.nodes.push(node);
        } else {
            if id >= self.nodes.len() {
                while self.nodes.len() <= id {
                    self.nodes.push(HnswNode {
                        id: self.nodes.len(),
                        vector: Vec::new(),
                        connections: Vec::new(),
                        level: 0,
                    });
                }
            }
            self.nodes[id] = node;
        }

        if self.entry_point == -1 {
            self.entry_point = id as isize;
            self.max_level = level as isize;
            return;
        }

        let mut current = self.entry_point as usize;
        for l in (level + 1..=self.max_level as usize).rev() {
            let best = self.search_layer(&self.nodes[id].vector, current, 1, l);
            if let Some(first) = best.first() {
                current = first.id;
            }
        }

        let max_connect_level = std::cmp::min(level, self.max_level as usize);
        for l in (0..=max_connect_level).rev() {
            let candidates = self.search_layer(&self.nodes[id].vector, current, self.ef_construction, l);
            let neighbors = self.select_neighbor_ids(&candidates, self.m);
            self.nodes[id].connections[l] = neighbors.clone();
            for neighbor_id in neighbors {
                self.connect(neighbor_id, id, l);
            }
            if let Some(first) = candidates.first() {
                current = first.id;
            }
        }

        if level as isize > self.max_level {
            self.entry_point = id as isize;
            self.max_level = level as isize;
        }
    }

    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<ResultItem> {
        if self.entry_point == -1 || self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut ef = if ef == 0 { self.ef_search } else { ef };
        if ef < k {
            ef = k;
        }

        let mut current = self.entry_point as usize;
        for l in (1..=self.max_level as usize).rev() {
            let best = self.search_layer(query, current, 1, l);
            if let Some(first) = best.first() {
                current = first.id;
            }
        }

        let mut candidates = self.search_layer(query, current, ef, 0);
        candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        candidates.truncate(k);
        candidates
            .into_iter()
            .map(|c| ResultItem { id: c.id, dist: c.dist })
            .collect()
    }

    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowed: &RoaringBitmap,
    ) -> Vec<ResultItem> {
        if allowed.is_empty() || self.entry_point == -1 || self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut ef = if ef == 0 { self.ef_search } else { ef };
        if ef < k {
            ef = k;
        }

        let mut entry = self.entry_point as usize;
        if !allowed.contains(entry as u32) {
            if let Some(first) = allowed.iter().next() {
                entry = first as usize;
            } else {
                return Vec::new();
            }
        }

        let mut current = entry;
        for l in (1..=self.max_level as usize).rev() {
            let best = self.search_layer_filtered(query, current, 1, l, allowed);
            if let Some(first) = best.first() {
                current = first.id;
            }
        }

        let mut candidates = self.search_layer_filtered(query, current, ef, 0, allowed);
        candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        candidates.truncate(k);
        candidates
            .into_iter()
            .map(|c| ResultItem { id: c.id, dist: c.dist })
            .collect()
    }

    pub fn marshal_binary(&self) -> Result<Vec<u8>, String> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TPHW");
        buf.write_u32::<LittleEndian>(1).map_err(|e| e.to_string())?;
        buf.write_u32::<LittleEndian>(self.nodes.len() as u32)
            .map_err(|e| e.to_string())?;
        buf.write_i32::<LittleEndian>(self.entry_point as i32)
            .map_err(|e| e.to_string())?;
        buf.write_i32::<LittleEndian>(self.max_level as i32)
            .map_err(|e| e.to_string())?;
        buf.write_u32::<LittleEndian>(self.m as u32)
            .map_err(|e| e.to_string())?;
        buf.write_u32::<LittleEndian>(self.ef_construction as u32)
            .map_err(|e| e.to_string())?;
        let metric = self.metric.as_str().as_bytes();
        buf.write_u32::<LittleEndian>(metric.len() as u32)
            .map_err(|e| e.to_string())?;
        buf.extend_from_slice(metric);

        for node in &self.nodes {
            buf.write_u32::<LittleEndian>(node.id as u32)
                .map_err(|e| e.to_string())?;
            buf.write_u32::<LittleEndian>(node.level as u32)
                .map_err(|e| e.to_string())?;
            for lvl in 0..=node.level {
                let conns = &node.connections[lvl];
                buf.write_u32::<LittleEndian>(conns.len() as u32)
                    .map_err(|e| e.to_string())?;
                for id in conns {
                    buf.write_u32::<LittleEndian>(*id as u32)
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(buf)
    }

    pub fn load_binary(data: &[u8], vectors: &[Vec<f32>], ef_search: usize) -> Result<Self, String> {
        let mut cursor = Cursor::new(data);
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"TPHW" {
            return Err("invalid HNSW format".to_string());
        }
        let version = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        if version != 1 {
            return Err(format!("unsupported HNSW version: {}", version));
        }
        let node_count = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
        let entry_point = cursor.read_i32::<LittleEndian>().map_err(|e| e.to_string())?;
        let max_level = cursor.read_i32::<LittleEndian>().map_err(|e| e.to_string())?;
        let m = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
        let ef_construction = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
        let metric_len = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
        let mut metric_bytes = vec![0u8; metric_len];
        cursor.read_exact(&mut metric_bytes).map_err(|e| e.to_string())?;
        let metric_str = std::str::from_utf8(&metric_bytes).map_err(|e| e.to_string())?;
        let metric = DistanceMetric::parse(Some(metric_str));

        if node_count != vectors.len() {
            return Err(format!(
                "vector count mismatch: index has {}, segment has {}",
                node_count,
                vectors.len()
            ));
        }

        let mut index = Self::new(m, ef_construction, ef_search, metric);
        index.entry_point = entry_point as isize;
        index.max_level = max_level as isize;
        index.nodes = vec![HnswNode {
            id: 0,
            vector: Vec::new(),
            connections: Vec::new(),
            level: 0,
        }; node_count];

        for _ in 0..node_count {
            let node_id = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
            let level = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
            if node_id >= vectors.len() {
                return Err(format!("node id {} out of range", node_id));
            }
            let mut node = HnswNode {
                id: node_id,
                vector: vectors[node_id].clone(),
                level,
                connections: vec![Vec::new(); level + 1],
            };
            for lvl in 0..=level {
                let conn_count = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
                let mut conns = Vec::with_capacity(conn_count);
                for _ in 0..conn_count {
                    let conn_id = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
                    conns.push(conn_id);
                }
                node.connections[lvl] = conns;
            }
            let id = node.id;
            index.nodes[id] = node;
        }

        Ok(index)
    }

    fn random_level(&mut self) -> usize {
        let p: f64 = self.rng.gen::<f64>().max(f64::MIN_POSITIVE);
        (-p.ln() * self.ml) as usize
    }

    fn search_layer(&self, query: &[f32], entry: usize, ef: usize, level: usize) -> Vec<ResultItem> {
        if entry >= self.nodes.len() {
            return Vec::new();
        }
        let ef = ef.max(1);

        let mut visited = vec![false; self.nodes.len()];
        let entry_dist = distance(query, &self.nodes[entry].vector, self.metric);

        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();

        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));
        results.push((OrderedFloat(entry_dist), entry));
        visited[entry] = true;

        while let Some(Reverse((OrderedFloat(current_dist), current))) = candidates.pop() {
            let worst = results.peek().map(|(dist, _)| dist.0).unwrap_or(f32::MAX);
            if current_dist > worst {
                break;
            }

            if level >= self.nodes[current].connections.len() {
                continue;
            }
            for &neighbor in &self.nodes[current].connections[level] {
                if neighbor >= self.nodes.len() {
                    continue;
                }
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;
                let dist = distance(query, &self.nodes[neighbor].vector, self.metric);
                if results.len() < ef || dist < results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX) {
                    candidates.push(Reverse((OrderedFloat(dist), neighbor)));
                    results.push((OrderedFloat(dist), neighbor));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out = Vec::with_capacity(results.len());
        while let Some((OrderedFloat(dist), id)) = results.pop() {
            out.push(ResultItem { id, dist });
        }
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        out
    }

    fn search_layer_filtered(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
        allowed: &RoaringBitmap,
    ) -> Vec<ResultItem> {
        if entry >= self.nodes.len() || !allowed.contains(entry as u32) {
            return Vec::new();
        }
        let ef = ef.max(1);

        let mut visited = vec![false; self.nodes.len()];
        let entry_dist = distance(query, &self.nodes[entry].vector, self.metric);

        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();

        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));
        results.push((OrderedFloat(entry_dist), entry));
        visited[entry] = true;

        while let Some(Reverse((OrderedFloat(current_dist), current))) = candidates.pop() {
            let worst = results.peek().map(|(dist, _)| dist.0).unwrap_or(f32::MAX);
            if current_dist > worst {
                break;
            }

            if level >= self.nodes[current].connections.len() {
                continue;
            }
            for &neighbor in &self.nodes[current].connections[level] {
                if neighbor >= self.nodes.len() {
                    continue;
                }
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;
                if !allowed.contains(neighbor as u32) {
                    continue;
                }
                let dist = distance(query, &self.nodes[neighbor].vector, self.metric);
                if results.len() < ef || dist < results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX) {
                    candidates.push(Reverse((OrderedFloat(dist), neighbor)));
                    results.push((OrderedFloat(dist), neighbor));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out = Vec::with_capacity(results.len());
        while let Some((OrderedFloat(dist), id)) = results.pop() {
            out.push(ResultItem { id, dist });
        }
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        out
    }

    fn select_neighbor_ids(&self, candidates: &[ResultItem], max_neighbors: usize) -> Vec<usize> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        sorted.truncate(max_neighbors);
        sorted.into_iter().map(|c| c.id).collect()
    }

    fn connect(&mut self, node_id: usize, neighbor_id: usize, level: usize) {
        if node_id >= self.nodes.len() || level >= self.nodes[node_id].connections.len() {
            return;
        }
        if self.nodes[node_id].connections[level].iter().any(|&id| id == neighbor_id) {
            return;
        }
        self.nodes[node_id].connections[level].push(neighbor_id);
        if self.nodes[node_id].connections[level].len() > self.m {
            let target = self.nodes[node_id].vector.clone();
            let mut candidates = Vec::new();
            for &id in &self.nodes[node_id].connections[level] {
                if id >= self.nodes.len() {
                    continue;
                }
                let dist = distance(&target, &self.nodes[id].vector, self.metric);
                candidates.push(ResultItem { id, dist });
            }
            self.nodes[node_id].connections[level] = self.select_neighbor_ids(&candidates, self.m);
        }
    }
}
