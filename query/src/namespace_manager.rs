use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tracing::{info, warn};

use tidepool_common::document::NamespaceInfo;
use tidepool_common::manifest::Manager as ManifestManager;
use tidepool_common::storage::{Store, StorageError};
use tidepool_common::wal::Reader as WalReader;

use crate::buffer::HotBuffer;
use crate::engine::{Engine, EngineOptions};

#[derive(Debug, thiserror::Error)]
pub enum NamespaceError {
    #[error("namespace not allowed")]
    NotAllowed,
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
}

struct NamespaceEntry<S: Store + Clone> {
    engine: Arc<Engine<S>>,
    last_access: Instant,
}

pub struct NamespaceManager<S: Store + Clone> {
    storage: S,
    cache_dir: Option<String>,
    options: EngineOptions,
    hot_buffer_max_size: usize,
    max_namespaces: Option<usize>,
    idle_timeout: Option<Duration>,
    allowed_namespaces: Option<HashSet<String>>,
    fixed_namespace: Option<String>,
    namespaces: RwLock<HashMap<String, NamespaceEntry<S>>>,
}

impl<S: Store + Clone + 'static> NamespaceManager<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        storage: S,
        cache_dir: Option<String>,
        options: EngineOptions,
        hot_buffer_max_size: usize,
        allowed_namespaces: Option<Vec<String>>,
        max_namespaces: Option<usize>,
        idle_timeout: Option<Duration>,
        fixed_namespace: Option<String>,
    ) -> Self {
        let allowed_namespaces = match fixed_namespace.as_ref() {
            Some(ns) => Some([ns.clone()].into_iter().collect()),
            None => allowed_namespaces.map(|vals| vals.into_iter().collect()),
        };

        Self {
            storage,
            cache_dir,
            options,
            hot_buffer_max_size,
            max_namespaces,
            idle_timeout,
            allowed_namespaces,
            fixed_namespace,
            namespaces: RwLock::new(HashMap::new()),
        }
    }

    pub fn fixed_namespace(&self) -> Option<&str> {
        self.fixed_namespace.as_deref()
    }

    pub async fn active_namespaces(&self) -> Vec<String> {
        let guard = self.namespaces.read().await;
        let mut names: Vec<String> = guard.keys().cloned().collect();
        names.sort();
        names
    }

    pub async fn get_engine(&self, namespace: &str) -> Result<Arc<Engine<S>>, NamespaceError> {
        if !self.is_allowed(namespace) {
            return Err(NamespaceError::NotAllowed);
        }

        self.evict_idle().await;

        let mut evicted_engines = Vec::new();

        let engine = {
            let mut guard = self.namespaces.write().await;
            if let Some(entry) = guard.get_mut(namespace) {
                entry.last_access = Instant::now();
                return Ok(entry.engine.clone());
            }

            if let Some(max_namespaces) = self.max_namespaces {
                if guard.len() >= max_namespaces {
                    if let Some(evicted) = evict_lru(&mut guard) {
                        evicted_engines.push(evicted);
                    }
                }
            }

            let hot_buffer = if self.hot_buffer_max_size > 0 {
                Some(Arc::new(HotBuffer::new(self.hot_buffer_max_size)))
            } else {
                None
            };

            let engine = Engine::new_with_buffer(
                self.storage.clone(),
                namespace.to_string(),
                self.cache_dir.clone(),
                self.options.clone(),
                hot_buffer,
            );
            let engine = Arc::new(engine);
            guard.insert(
                namespace.to_string(),
                NamespaceEntry {
                    engine: engine.clone(),
                    last_access: Instant::now(),
                },
            );
            engine
        };

        for evicted in evicted_engines {
            evicted.invalidate_cache().await;
        }

        if let Err(err) = engine.load_manifest().await {
            warn!("Failed to load manifest for namespace {}: {}", namespace, err);
        }

        Ok(engine)
    }

    pub async fn get_namespace_info(&self, namespace: &str) -> Result<NamespaceInfo, NamespaceError> {
        if !self.is_allowed(namespace) {
            return Err(NamespaceError::NotAllowed);
        }

        let manifest_manager = ManifestManager::new(self.storage.clone(), namespace);
        match manifest_manager.load().await {
            Ok(manifest) => Ok(NamespaceInfo {
                namespace: namespace.to_string(),
                approx_count: manifest.total_doc_count(),
                dimensions: manifest.dimensions,
                pending_compaction: None,
            }),
            Err(StorageError::NotFound(_)) => {
                let wal_reader = WalReader::new(self.storage.clone(), namespace);
                let wal_files = wal_reader.list_wal_files().await?;
                let pending = if wal_files.is_empty() { None } else { Some(true) };
                Ok(NamespaceInfo {
                    namespace: namespace.to_string(),
                    approx_count: 0,
                    dimensions: 0,
                    pending_compaction: pending,
                })
            }
            Err(err) => Err(NamespaceError::Storage(err)),
        }
    }

    pub async fn list_namespaces(&self) -> Result<Vec<NamespaceInfo>, NamespaceError> {
        let mut names = HashSet::new();
        if let Some(ns) = self.fixed_namespace.as_ref() {
            names.insert(ns.clone());
        }

        let keys = self.storage.list("namespaces/").await?;
        for key in keys {
            if let Some(ns) = parse_namespace_from_key(&key) {
                names.insert(ns.to_string());
            }
        }

        let mut infos = Vec::new();
        for ns in names {
            if !self.is_allowed(&ns) {
                continue;
            }
            match self.get_namespace_info(&ns).await {
                Ok(info) => infos.push(info),
                Err(NamespaceError::NotAllowed) => {}
                Err(err) => return Err(err),
            }
        }

        infos.sort_by(|a, b| a.namespace.cmp(&b.namespace));
        Ok(infos)
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
                    evicted.push((name, entry.engine));
                }
            }
        }

        for (name, engine) in evicted {
            engine.invalidate_cache().await;
            info!("Evicted idle namespace {}", name);
        }
    }
}

fn evict_lru<S: Store + Clone>(
    guard: &mut HashMap<String, NamespaceEntry<S>>,
) -> Option<Arc<Engine<S>>> {
    let lru_name = guard
        .iter()
        .min_by_key(|(_, entry)| entry.last_access)
        .map(|(name, _)| name.clone())?;
    guard.remove(&lru_name).map(|entry| entry.engine)
}

fn parse_namespace_from_key(key: &str) -> Option<&str> {
    let rest = key.strip_prefix("namespaces/")?;
    let ns = rest.split('/').next()?;
    if ns.is_empty() {
        None
    } else {
        Some(ns)
    }
}
