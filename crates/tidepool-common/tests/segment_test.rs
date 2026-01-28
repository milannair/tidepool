use std::collections::BTreeMap;

use tidepool_common::attributes::AttrValue;
use tidepool_common::document::Document;
use tidepool_common::segment::{Reader, Writer};
use tidepool_common::storage::InMemoryStore;

fn tag_attr(value: &str) -> AttrValue {
    let mut map = BTreeMap::new();
    map.insert("tag".to_string(), AttrValue::String(value.to_string()));
    AttrValue::Object(map)
}

#[tokio::test]
async fn segment_roundtrip() {
    let store = InMemoryStore::new();
    let writer = Writer::new(store.clone(), "test");
    let reader = Reader::new(store.clone(), "test", None);

    let docs = vec![
        Document {
            id: "a".to_string(),
            vector: vec![1.0, 2.0],
            attributes: Some(tag_attr("x")),
        },
        Document {
            id: "b".to_string(),
            vector: vec![3.0, 4.0],
            attributes: Some(tag_attr("y")),
        },
    ];

    let seg = writer.write_segment(&docs).await.unwrap().unwrap();
    let loaded = reader.read_segment(&seg.segment_key).await.unwrap();

    assert_eq!(loaded.dimensions, 2);
    assert_eq!(loaded.ids.len(), 2);
    assert_eq!(loaded.ids[0], "a");
    assert_eq!(loaded.ids[1], "b");
    assert_eq!(loaded.vectors[0][0], 1.0);
    assert_eq!(loaded.vectors[1][1], 4.0);
    assert!(loaded.index.is_some());
}

#[tokio::test]
async fn segment_dimension_mismatch() {
    let store = InMemoryStore::new();
    let writer = Writer::new(store, "test");

    let docs = vec![
        Document {
            id: "a".to_string(),
            vector: vec![1.0, 2.0],
            attributes: None,
        },
        Document {
            id: "b".to_string(),
            vector: vec![3.0],
            attributes: None,
        },
    ];

    let result = writer.write_segment(&docs).await;
    assert!(result.is_err());
}
