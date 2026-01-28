use serde_json::json;

use tidepool_common::document::{Document, QueryRequest};
use tidepool_common::manifest::{Manager, Manifest, Segment};
use tidepool_common::segment::Writer;
use tidepool_common::storage::InMemoryStore;

use tidepool_query::engine::Engine;

#[tokio::test]
async fn engine_query_with_filters() {
    let store = InMemoryStore::new();
    let namespace = "test";

    let writer = Writer::new(store.clone(), namespace);
    let docs = vec![
        Document {
            id: "a".to_string(),
            vector: vec![1.0, 0.0],
            attributes: Some(json!({"tag": "x"})),
        },
        Document {
            id: "b".to_string(),
            vector: vec![0.0, 1.0],
            attributes: Some(json!({"tag": "y"})),
        },
    ];

    let seg = writer.write_segment(&docs).await.unwrap().unwrap();
    let manifest = Manifest::new(vec![Segment {
        id: seg.id.clone(),
        segment_key: seg.segment_key.clone(),
        doc_count: seg.doc_count,
        dimensions: seg.dimensions,
    }]);
    let manager = Manager::new(store.clone(), namespace);
    manager.save(&manifest).await.unwrap();

    let engine = Engine::new(store.clone(), namespace.to_string(), None);
    engine.load_manifest().await.unwrap();

    let req = QueryRequest {
        vector: vec![1.0, 0.0],
        top_k: 1,
        ef_search: 0,
        distance_metric: None,
        include_vectors: false,
        filters: Some(json!({"tag": "x"})),
    };

    let resp = engine.query(&req).await.unwrap();
    assert_eq!(resp.results.len(), 1);
    assert_eq!(resp.results[0].id, "a");
}
