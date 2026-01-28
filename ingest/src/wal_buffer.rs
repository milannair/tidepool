use std::time::Duration;

use chrono::Utc;
use tokio::sync::{mpsc, oneshot};

use tidepool_common::document::Document;
use tidepool_common::storage::{Store, StorageError};
use tidepool_common::wal::{Entry, Writer as WalWriter};
use tidepool_common::document::RkyvDocument;

struct WalRequest {
    entries: Vec<Entry>,
    respond_to: oneshot::Sender<Result<(), StorageError>>,
}

#[derive(Clone)]
pub struct BufferedWalWriter<S: Store + Clone + Send + Sync + 'static> {
    sender: mpsc::Sender<WalRequest>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Store + Clone + Send + Sync + 'static> BufferedWalWriter<S> {
    pub fn new(writer: WalWriter<S>, max_entries: usize, flush_interval: Duration) -> Self {
        let max_entries = max_entries.max(1);
        let (sender, mut receiver) = mpsc::channel::<WalRequest>(max_entries * 4);
        let flush_every = if flush_interval.is_zero() {
            Duration::from_secs(3600)
        } else {
            flush_interval
        };

        tokio::spawn(async move {
            let mut buffer: Vec<WalRequest> = Vec::new();
            let mut entry_count = 0usize;
            let mut ticker = tokio::time::interval(flush_every);

            loop {
                tokio::select! {
                    maybe_req = receiver.recv() => {
                        match maybe_req {
                            Some(req) => {
                                entry_count += req.entries.len();
                                buffer.push(req);
                                if max_entries > 0 && entry_count >= max_entries {
                                    flush_batch(&writer, &mut buffer, &mut entry_count).await;
                                }
                            }
                            None => {
                                if !buffer.is_empty() {
                                    flush_batch(&writer, &mut buffer, &mut entry_count).await;
                                }
                                break;
                            }
                        }
                    }
                    _ = ticker.tick() => {
                        if !buffer.is_empty() {
                            flush_batch(&writer, &mut buffer, &mut entry_count).await;
                        }
                    }
                }
            }
        });

        Self {
            sender,
            _phantom: std::marker::PhantomData,
        }
    }

    pub async fn write_upsert(&self, docs: Vec<Document>) -> Result<Option<String>, StorageError> {
        if docs.is_empty() {
            return Ok(None);
        }

        let entries: Vec<Entry> = docs
            .into_iter()
            .map(|doc| Entry {
                ts: now_ts_nanos(),
                op: "upsert".to_string(),
                doc: Some(RkyvDocument::from(&doc)),
                delete_ids: Vec::new(),
            })
            .collect();

        self.send_entries(entries).await?;
        Ok(None)
    }

    pub async fn write_delete(&self, ids: Vec<String>) -> Result<Option<String>, StorageError> {
        if ids.is_empty() {
            return Ok(None);
        }

        let entry = Entry {
            ts: now_ts_nanos(),
            op: "delete".to_string(),
            doc: None,
            delete_ids: ids,
        };

        self.send_entries(vec![entry]).await?;
        Ok(None)
    }

    async fn send_entries(&self, entries: Vec<Entry>) -> Result<(), StorageError> {
        let (respond_to, response) = oneshot::channel();
        let req = WalRequest { entries, respond_to };
        self.sender
            .send(req)
            .await
            .map_err(|_| StorageError::Other("wal batch channel closed".to_string()))?;
        response
            .await
            .map_err(|_| StorageError::Other("wal batch response dropped".to_string()))?
    }
}

async fn flush_batch<S: Store>(
    writer: &WalWriter<S>,
    buffer: &mut Vec<WalRequest>,
    entry_count: &mut usize,
) {
    if buffer.is_empty() {
        return;
    }

    let mut entries = Vec::with_capacity(*entry_count);
    let mut responders = Vec::with_capacity(buffer.len());
    for req in buffer.drain(..) {
        *entry_count = entry_count.saturating_sub(req.entries.len());
        entries.extend(req.entries);
        responders.push(req.respond_to);
    }

    let result = writer.write_entries(&entries).await.map(|_| ());
    for responder in responders {
        let _ = responder.send(result.as_ref().map(|_| ()).map_err(clone_error));
    }
}

fn clone_error(err: &StorageError) -> StorageError {
    match err {
        StorageError::NotFound(value) => StorageError::NotFound(value.clone()),
        StorageError::Other(value) => StorageError::Other(value.clone()),
    }
}

fn now_ts_nanos() -> i64 {
    Utc::now()
        .timestamp_nanos_opt()
        .unwrap_or_else(|| Utc::now().timestamp_millis() * 1_000_000)
}
