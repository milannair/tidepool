//! Pub/Sub invalidation listener for real-time cache invalidation.
//!
//! This module provides a background task that listens for compaction events
//! via Redis pub/sub and invalidates the local query cache accordingly.

use std::sync::Arc;

use futures_util::StreamExt;
use tracing::{debug, error, info, warn};

use tidepool_common::redis::RedisStore;
use tidepool_common::storage::Store;

use crate::engine::Engine;

/// Background task that listens for invalidation events and updates the engine.
pub struct InvalidationListener<S: Store + Clone + 'static> {
    redis: Arc<RedisStore>,
    engine: Arc<Engine<S>>,
    namespace: String,
}

impl<S: Store + Clone + 'static> InvalidationListener<S> {
    pub fn new(redis: Arc<RedisStore>, engine: Arc<Engine<S>>, namespace: String) -> Self {
        Self {
            redis,
            engine,
            namespace,
        }
    }

    /// Start listening for invalidation events.
    /// This runs indefinitely until the task is cancelled.
    pub async fn run(&self) {
        info!(
            "Starting invalidation listener for namespace {}",
            self.namespace
        );

        loop {
            match self.listen_once().await {
                Ok(()) => {
                    // Connection closed gracefully, reconnect
                    info!("Invalidation listener connection closed, reconnecting...");
                }
                Err(e) => {
                    error!("Invalidation listener error: {}, reconnecting in 5s...", e);
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        }
    }

    async fn listen_once(&self) -> Result<(), String> {
        let mut pubsub = self
            .redis
            .subscribe_invalidations(&self.namespace)
            .await
            .map_err(|e| format!("Failed to subscribe: {}", e))?;

        let mut stream = pubsub.on_message();

        while let Some(msg) = stream.next().await {
            let payload: String = msg
                .get_payload()
                .map_err(|e| format!("Failed to get payload: {}", e))?;

            debug!("Received invalidation message: {}", payload);

            if let Ok(event) = serde_json::from_str::<serde_json::Value>(&payload) {
                let event_type = event
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");

                match event_type {
                    "compaction_complete" => {
                        info!(
                            "Received compaction_complete event for namespace {}",
                            self.namespace
                        );

                        // Invalidate local cache and reload manifest
                        self.engine.invalidate_cache().await;

                        // Trigger a reload to load new segments
                        if let Err(e) = self.engine.reload_segments().await {
                            warn!("Failed to reload segments after compaction: {}", e);
                        }

                        info!("Cache invalidated and reloaded after compaction");
                    }
                    other => {
                        debug!("Ignoring unknown event type: {}", other);
                    }
                }
            }
        }

        Ok(())
    }
}

/// Spawn an invalidation listener as a background task.
pub fn spawn_invalidation_listener<S: Store + Clone + Send + Sync + 'static>(
    redis: Arc<RedisStore>,
    engine: Arc<Engine<S>>,
    namespace: String,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let listener = InvalidationListener::new(redis, engine, namespace);
        listener.run().await;
    })
}
