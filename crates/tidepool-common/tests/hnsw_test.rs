use tidepool_common::index::hnsw::HnswIndex;
use tidepool_common::vector::DistanceMetric;

#[test]
fn hnsw_roundtrip() {
    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut index = HnswIndex::new(4, 32, 16, DistanceMetric::Cosine);
    index.set_seed(42);
    for (i, vec) in vectors.iter().cloned().enumerate() {
        index.insert(i, vec);
    }

    let data = index.marshal_binary().unwrap();
    let loaded = HnswIndex::load_binary(&data, &vectors, 16).unwrap();

    let results = loaded.search(&[1.0, 0.0], 1, 0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
}
