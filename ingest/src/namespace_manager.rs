use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::info;

use tidepool_common::redis::RedisStore;
use tidepool_common::segment::WriterOptions;
use tidepool_common::storage::{Store, StorageError};
use tidepool_common::wal::Writer as WalWriter;

use crate::compactor::Compactor;
use crate::wal_buffer::BufferedWalWriter;

#[derive(Debug, thiserror::Error)]
pub enum NamespaceError {
    #[error("namespace not allowed")]
    NotAllowed,
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
}

#[derive(Clone)]
pub struct NamespaceHandle<S: Store + Clone + Send + Sync + 'static> {
    pub wal_writer: Arc<BufferedWalWriter<S>>,
    pub compactor: Arc<Compactor<S>>,
}

struct NamespaceEntry<S: Store + Clone + Send + Sync + 'static> {
    handle: NamespaceHandle<S>,
    compactor_task: JoinHandle<()>,
    last_access: Instant,
}

pub struct NamespaceManager<S: Store + Clone + Send + Sync + 'static> {
    storage: S,
    writer_options: WriterOptions,
    wal_batch_max_entries: usize,
    wal_batch_flush_interval: Duration,
    compaction_interval: Duration,
    max_namespaces: Option<usize>,
    idle_timeout: Option<Duration>,
    allowed_namespaces: Option<HashSet<String>>,
    fixed_namespace: Option<String>,
    redis: Option<Arc<RedisStore>>,
    namespaces: RwLock<HashMap<String, NamespaceEntry<S>>>,
}

impl<S: Store + Clone + Send + Sync + 'static> NamespaceManager<S> {
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn new(
        storage: S,
        writer_options: WriterOptions,
        wal_batch_max_entries: usize,
        wal_batch_flush_interval: Duration,
        compaction_interval: Duration,
        allowed_namespaces: Option<Vec<String>>,
        max_namespaces: Option<usize>,
        idle_timeout: Option<Duration>,
        fixed_namespace: Option<String>,
    ) -> Self {
        Self::new_with_redis(
            storage,
            writer_options,
            wal_batch_max_entries,
            wal_batch_flush_interval,
            compaction_interval,
            allowed_namespaces,
            max_namespaces,
            idle_timeout,
            fixed_namespace,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_redis(
        storage: S,
        writer_options: WriterOptions,
        wal_batch_max_entries: usize,
        wal_batch_flush_interval: Duration,
        compaction_interval: Duration,
        allowed_namespaces: Option<Vec<String>>,
        max_namespaces: Option<usize>,
        idle_timeout: Option<Duration>,
        fixed_namespace: Option<String>,
        redis: Option<Arc<RedisStore>>,
    ) -> Self {
        let allowed_namespaces = match fixed_namespace.as_ref() {
            Some(ns) => Some([ns.clone()].into_iter().collect()),
            None => allowed_namespaces.map(|vals| vals.into_iter().collect()),
        };

        Self {
            storage,
            writer_options,
            wal_batch_max_entries,
            wal_batch_flush_interval,
            compaction_interval,
            max_namespaces,
            idle_timeout,
            allowed_namespaces,
            fixed_namespace,
            redis,
            namespaces: RwLock::new(HashMap::new()),
        }
    }

    pub fn fixed_namespace(&self) -> Option<&str> {
        self.fixed_namespace.as_deref()
    }

    pub async fn get_or_create(&self, namespace: &str) -> Result<NamespaceHandle<S>, NamespaceError> {
        if !self.is_allowed(namespace) {
            return Err(NamespaceError::NotAllowed);
        }

        self.evict_idle().await;

        let mut evicted = Vec::new();
        let handle = {
            let mut guard = self.namespaces.write().await;
            if let Some(entry) = guard.get_mut(namespace) {
                entry.last_access = Instant::now();
                return Ok(entry.handle.clone());
            }

            if let Some(max_namespaces) = self.max_namespaces {
                if guard.len() >= max_namespaces {
                    if let Some(evicted_entry) = evict_lru(&mut guard) {
                        evicted.push(evicted_entry);
                    }
                }
            }

            let wal_writer = Arc::new(BufferedWalWriter::new_with_redis(
                WalWriter::new(self.storage.clone(), namespace.to_string()),
                self.wal_batch_max_entries,
                self.wal_batch_flush_interval,
                self.redis.clone(),
                namespace.to_string(),
            ));
            let compactor = Arc::new(Compactor::new_with_redis(
                self.storage.clone(),
                namespace.to_string(),
                self.writer_options.clone(),
                self.redis.clone(),
            ));

            let compactor_task = {
                let compactor = compactor.clone();
                let interval = self.compaction_interval;
                tokio::spawn(async move {
                    compactor.run_periodically(interval).await;
                })
            };

            let handle = NamespaceHandle { wal_writer, compactor };
            guard.insert(
                namespace.to_string(),
                NamespaceEntry {
                    handle: handle.clone(),
                    compactor_task,
                    last_access: Instant::now(),
                },
            );
            handle
        };

        for entry in evicted {
            entry.compactor_task.abort();
            info!("Evicted namespace {}", entry.name);
        }

        Ok(handle)
    }

    pub async fn shutdown(&self) {
        let mut entries = Vec::new();
        {
            let mut guard = self.namespaces.write().await;
            for (name, entry) in guard.drain() {
                entries.push((name, entry));
            }
        }

        for (name, entry) in entries {
            entry.compactor_task.abort();
            info!("Stopped compactor for namespace {}", name);
        }
    }

    fn is_allowed(&self, namespace: &str) -> bool {
        match &self.allowed_namespaces {
            Some(allowed) => allowed.contains(namespace),
            None => true,
        }
    }

    async fn evict_idle(&self) {
        let Some(timeout) = self.idle_timeout else {
            return;
        };

        let mut evicted = Vec::new();
        {
            let mut guard = self.namespaces.write().await;
            let stale: Vec<String> = guard
                .iter()
                .filter(|(_, entry)| entry.last_access.elapsed() > timeout)
                .map(|(name, _)| name.clone())
                .collect();
            for name in stale {
                if let Some(entry) = guard.remove(&name) {
                    evicted.push((name, entry));
                }
            }
        }

        for (name, entry) in evicted {
            entry.compactor_task.abort();
            info!("Evicted idle namespace {}", name);
        }
    }
}

struct EvictedEntry<S: Store + Clone + Send + Sync + 'static> {
    name: String,
    compactor_task: JoinHandle<()>,
    _handle: NamespaceHandle<S>,
}

fn evict_lru<S: Store + Clone + Send + Sync + 'static>(
    guard: &mut HashMap<String, NamespaceEntry<S>>,
) -> Option<EvictedEntry<S>> {
    let lru_name = guard
        .iter()
        .min_by_key(|(_, entry)| entry.last_access)
        .map(|(name, _)| name.clone())?;
    guard.remove(&lru_name).map(|entry| EvictedEntry {
        name: lru_name,
        compactor_task: entry.compactor_task,
        _handle: entry.handle,
    })
}
