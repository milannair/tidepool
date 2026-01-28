use std::collections::HashSet;

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

use crate::storage::{tombstone_path, Store, StorageError};

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
struct TombstoneSet {
    ids: Vec<String>,
}

#[derive(Clone)]
pub struct Manager<S: Store> {
    storage: S,
    namespace: String,
}

impl<S: Store> Manager<S> {
    pub fn new(storage: S, namespace: impl Into<String>) -> Self {
        Self {
            storage,
            namespace: namespace.into(),
        }
    }

    pub async fn load(&self) -> Result<HashSet<String>, StorageError> {
        let data = match self.storage.get(&tombstone_path(&self.namespace)).await {
            Ok(data) => data,
            Err(StorageError::NotFound(_)) => return Ok(HashSet::new()),
            Err(err) => return Err(err),
        };
        // SAFETY: We trust our own serialized data format
        let archived = unsafe { rkyv::archived_root::<TombstoneSet>(&data) };
        let tombstones: TombstoneSet = archived
            .deserialize(&mut rkyv::Infallible)
            .map_err(|err| StorageError::Other(format!("parse tombstones: {}", err)))?;
        Ok(tombstones.ids.into_iter().collect())
    }

    pub async fn save(&self, tombstones: &HashSet<String>) -> Result<(), StorageError> {
        let mut ids: Vec<String> = tombstones.iter().cloned().collect();
        ids.sort();
        let data = rkyv::to_bytes::<_, 256>(&TombstoneSet { ids })
            .map_err(|err| StorageError::Other(format!("serialize tombstones: {}", err)))?
            .as_ref()
            .to_vec();
        self.storage.put(&tombstone_path(&self.namespace), data).await
    }
}
