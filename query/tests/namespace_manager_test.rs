use tidepool_common::document::Document;
use tidepool_common::manifest::{Manager, Manifest, Segment};
use tidepool_common::segment::Writer;
use tidepool_common::storage::InMemoryStore;
use tidepool_common::wal::Writer as WalWriter;

use tidepool_query::engine::EngineOptions;
use tidepool_query::namespace_manager::NamespaceManager;

#[tokio::test]
async fn namespace_info_pending_compaction() {
    let store = InMemoryStore::new();
    let wal_writer = WalWriter::new(store.clone(), "pending");
    wal_writer
        .write_upsert(vec![Document {
            id: "doc-1".to_string(),
            vector: vec![1.0, 2.0],
            text: None,
            attributes: None,
        }])
        .await
        .unwrap();

    let manager = NamespaceManager::new(
        store.clone(),
        None,
        EngineOptions::default(),
        0,
        None,
        None,
        None,
        None,
    );

    let info = manager.get_namespace_info("pending").await.unwrap();
    assert_eq!(info.namespace, "pending");
    assert_eq!(info.approx_count, 0);
    assert_eq!(info.dimensions, 0);
    assert_eq!(info.pending_compaction, Some(true));
}

#[tokio::test]
async fn list_namespaces_includes_manifest_and_wal() {
    let store = InMemoryStore::new();

    let writer = Writer::new(store.clone(), "ready");
    let seg = writer
        .write_segment(&[Document {
            id: "doc-1".to_string(),
            vector: vec![1.0],
            text: None,
            attributes: None,
        }])
        .await
        .unwrap()
        .unwrap();

    let manifest = Manifest::new(vec![Segment {
        id: seg.id.clone(),
        segment_key: seg.segment_key.clone(),
        doc_count: seg.doc_count,
        dimensions: seg.dimensions,
        size_bytes: seg.size_bytes,
        content_hash: Some(seg.content_hash.clone()),
        bloom_key: Some(seg.bloom_key.clone()),
    }]);
    let manager = Manager::new(store.clone(), "ready");
    manager.save(&manifest).await.unwrap();

    let wal_writer = WalWriter::new(store.clone(), "pending");
    wal_writer
        .write_upsert(vec![Document {
            id: "doc-2".to_string(),
            vector: vec![2.0],
            text: None,
            attributes: None,
        }])
        .await
        .unwrap();

    let manager = NamespaceManager::new(
        store.clone(),
        None,
        EngineOptions::default(),
        0,
        None,
        None,
        None,
        None,
    );

    let infos = manager.list_namespaces().await.unwrap();
    let mut names: Vec<String> = infos.iter().map(|i| i.namespace.clone()).collect();
    names.sort();
    assert_eq!(names, vec!["pending".to_string(), "ready".to_string()]);

    let pending = infos.iter().find(|i| i.namespace == "pending").unwrap();
    assert_eq!(pending.pending_compaction, Some(true));
}

#[tokio::test]
async fn lru_eviction_removes_oldest_namespace() {
    let store = InMemoryStore::new();
    let manager = NamespaceManager::new(
        store.clone(),
        None,
        EngineOptions::default(),
        0,
        None,
        Some(1),
        None,
        None,
    );

    let _ = manager.get_engine("alpha").await.unwrap();
    let active = manager.active_namespaces().await;
    assert_eq!(active, vec!["alpha".to_string()]);

    let _ = manager.get_engine("beta").await.unwrap();
    let active = manager.active_namespaces().await;
    assert_eq!(active, vec!["beta".to_string()]);
}
