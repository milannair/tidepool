//! Bloom filter for fast segment pruning during queries.
//!
//! Each segment has a Bloom filter containing all document IDs. This allows
//! the query engine to quickly reject segments that definitely don't contain
//! a queried ID, without loading the full segment data.

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use std::hash::{Hash, Hasher};

/// A simple Bloom filter optimized for segment ID lookups.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
pub struct BloomFilter {
    /// Bit array stored as bytes.
    bits: Vec<u8>,
    /// Number of hash functions to use.
    num_hashes: u32,
    /// Number of items inserted.
    count: u64,
}

impl BloomFilter {
    /// Create a new Bloom filter with the given capacity and false positive rate.
    ///
    /// # Arguments
    /// * `expected_items` - Expected number of items to insert
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let fp_rate = false_positive_rate.max(0.0001).min(0.5);
        let n = expected_items.max(1) as f64;

        // Optimal number of bits: m = -n * ln(p) / (ln(2)^2)
        let m = (-n * fp_rate.ln() / (2.0_f64.ln().powi(2))).ceil() as usize;
        let m = m.max(64); // Minimum 64 bits

        // Optimal number of hash functions: k = (m/n) * ln(2)
        let k = ((m as f64 / n) * 2.0_f64.ln()).ceil() as u32;
        let k = k.clamp(1, 16); // Between 1 and 16 hash functions

        let num_bytes = (m + 7) / 8;

        Self {
            bits: vec![0u8; num_bytes],
            num_hashes: k,
            count: 0,
        }
    }

    /// Create a Bloom filter with default settings for segment IDs.
    /// Assumes ~100K IDs per segment with 1% false positive rate.
    pub fn for_segment(expected_ids: usize) -> Self {
        Self::new(expected_ids.max(1000), 0.01)
    }

    /// Insert an item into the Bloom filter.
    pub fn insert<T: Hash>(&mut self, item: &T) {
        let (h1, h2) = self.hash_pair(item);
        let num_bits = self.bits.len() * 8;

        for i in 0..self.num_hashes {
            let idx = self.get_index(h1, h2, i, num_bits);
            self.set_bit(idx);
        }
        self.count += 1;
    }

    /// Check if an item might be in the set.
    /// Returns `true` if the item might be present (could be false positive).
    /// Returns `false` if the item is definitely not present.
    pub fn may_contain<T: Hash>(&self, item: &T) -> bool {
        let (h1, h2) = self.hash_pair(item);
        let num_bits = self.bits.len() * 8;

        for i in 0..self.num_hashes {
            let idx = self.get_index(h1, h2, i, num_bits);
            if !self.get_bit(idx) {
                return false;
            }
        }
        true
    }

    /// Returns true if the filter is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the number of items inserted.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the size of the filter in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bits.len()
    }

    /// Serialize the Bloom filter to bytes using rkyv.
    pub fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 256>(self)
            .map(|b| b.to_vec())
            .unwrap_or_default()
    }

    /// Deserialize a Bloom filter from rkyv bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        rkyv::check_archived_root::<BloomFilter>(data)
            .ok()?
            .deserialize(&mut rkyv::Infallible)
            .ok()
    }

    fn hash_pair<T: Hash>(&self, item: &T) -> (u64, u64) {
        // Use two different hash functions based on FNV-1a
        let mut h1 = FnvHasher::new();
        item.hash(&mut h1);
        let hash1 = h1.finish();

        let mut h2 = FnvHasher::with_seed(0x517cc1b727220a95);
        item.hash(&mut h2);
        let hash2 = h2.finish();

        (hash1, hash2)
    }

    fn get_index(&self, h1: u64, h2: u64, i: u32, num_bits: usize) -> usize {
        // Double hashing: h(i) = h1 + i * h2
        let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
        (combined % num_bits as u64) as usize
    }

    fn set_bit(&mut self, idx: usize) {
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        if byte_idx < self.bits.len() {
            self.bits[byte_idx] |= 1 << bit_idx;
        }
    }

    fn get_bit(&self, idx: usize) -> bool {
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        if byte_idx < self.bits.len() {
            (self.bits[byte_idx] & (1 << bit_idx)) != 0
        } else {
            false
        }
    }
}

/// Simple FNV-1a hasher.
struct FnvHasher {
    state: u64,
}

impl FnvHasher {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self {
            state: Self::FNV_OFFSET,
        }
    }

    fn with_seed(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(Self::FNV_PRIME);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut bloom = BloomFilter::new(1000, 0.01);

        bloom.insert(&"test1");
        bloom.insert(&"test2");
        bloom.insert(&"test3");

        assert!(bloom.may_contain(&"test1"));
        assert!(bloom.may_contain(&"test2"));
        assert!(bloom.may_contain(&"test3"));

        // These should almost certainly not be present
        // (false positive rate is 1%)
        let mut false_positives = 0;
        for i in 0..100 {
            if bloom.may_contain(&format!("not_present_{}", i)) {
                false_positives += 1;
            }
        }
        // Allow up to 5% false positives in test (generous margin)
        assert!(false_positives < 5, "Too many false positives: {}", false_positives);
    }

    #[test]
    fn test_serialization() {
        let mut bloom = BloomFilter::for_segment(1000);
        for i in 0..100 {
            bloom.insert(&format!("id_{}", i));
        }

        let bytes = bloom.to_bytes();
        let restored = BloomFilter::from_bytes(&bytes).unwrap();

        assert_eq!(bloom.count(), restored.count());
        for i in 0..100 {
            assert!(restored.may_contain(&format!("id_{}", i)));
        }
    }

    #[test]
    fn test_segment_bloom() {
        let mut bloom = BloomFilter::for_segment(10000);
        for i in 0..10000 {
            bloom.insert(&format!("doc_{}", i));
        }

        // Size should be reasonable (< 20 KB for 10K items at 1% FP)
        assert!(bloom.size_bytes() < 20 * 1024);

        // All inserted items should be found
        for i in 0..10000 {
            assert!(bloom.may_contain(&format!("doc_{}", i)));
        }
    }
}
