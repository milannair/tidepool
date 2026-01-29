use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::signal;
use tower_http::cors::{Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tracing::{error, info};

use tidepool_common::config::Config;
use tidepool_common::document::{DeleteRequest, DeleteResponse, UpsertRequest, UpsertResponse};
use tidepool_common::segment::WriterOptions;
use tidepool_common::storage::S3Store;

mod compactor;
mod namespace_manager;
use namespace_manager::{NamespaceError, NamespaceManager};
mod wal_buffer;

#[derive(Clone)]
struct AppState {
    namespaces: Arc<NamespaceManager<S3Store>>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Starting tidepool-ingest service...");

    let cfg = match Config::from_env() {
        Ok(cfg) => cfg,
        Err(err) => {
            error!("Configuration error: {}", err);
            std::process::exit(1);
        }
    };

    let storage = match S3Store::new(&cfg).await {
        Ok(store) => store,
        Err(err) => {
            error!("Failed to initialize storage client: {}", err);
            std::process::exit(1);
        }
    };

    let writer_options = WriterOptions {
        hnsw_m: cfg.hnsw_m,
        hnsw_ef_construction: cfg.hnsw_ef_construction,
        hnsw_ef_search: cfg.hnsw_ef_search,
        metric: tidepool_common::vector::DistanceMetric::Cosine,
        use_v3_format: true,
        text_index_enabled: cfg.text_index_enabled,
        tokenizer_config: cfg.tokenizer_config(),
        ivf_enabled: cfg.ivf_enabled,
        ivf_min_segment_size: cfg.ivf_min_segment_size,
        ivf_k_factor: cfg.ivf_k_factor,
        ivf_min_k: cfg.ivf_min_k,
        ivf_max_k: cfg.ivf_max_k,
        ivf_nprobe_default: cfg.ivf_nprobe_default,
        quantization: cfg.quantization,
        ..WriterOptions::default()
    };

    let namespaces = Arc::new(NamespaceManager::new(
        storage,
        writer_options,
        cfg.wal_batch_max_entries,
        cfg.wal_batch_flush_interval,
        cfg.compaction_interval,
        cfg.allowed_namespaces.clone(),
        cfg.max_namespaces,
        cfg.namespace_idle_timeout,
        cfg.namespace.clone(),
    ));

    let state = AppState {
        namespaces: namespaces.clone(),
    };

    let cors = if cfg.cors_allow_origin == "*" {
        CorsLayer::new().allow_origin(Any)
    } else {
        let origin = cfg
            .cors_allow_origin
            .parse::<HeaderValue>()
            .expect("invalid CORS_ALLOW_ORIGIN");
        CorsLayer::new().allow_origin(origin)
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/compact", post(compact))
        .route("/v1/namespaces/:namespace", post(upsert).delete(delete_vectors))
        .route("/v1/namespaces/:namespace/status", get(status_namespace))
        .route("/v1/namespaces/:namespace/compact", post(compact_namespace))
        .route("/v1/vectors/:namespace", post(upsert).delete(delete_vectors))
        .layer(cors)
        .layer(RequestBodyLimitLayer::new(cfg.max_body_bytes))
        .with_state(state);

    let addr: SocketAddr = format!("0.0.0.0:{}", cfg.port)
        .parse()
        .expect("invalid port");

    info!("Listening on {}", addr);

    let server = axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app);
    tokio::select! {
        _ = server => {},
        _ = signal::ctrl_c() => {
            info!("Shutting down...");
        }
    }
    namespaces.shutdown().await;
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "tidepool-ingest"
    }))
}

async fn status(State(state): State<AppState>) -> Response {
    let Some(namespace) = state.namespaces.fixed_namespace() else {
        return json_error(StatusCode::BAD_REQUEST, "namespace required");
    };
    status_namespace(Path(namespace.to_string()), State(state)).await
}

async fn compact(State(state): State<AppState>) -> Response {
    let _ = state;
    json_error(StatusCode::BAD_REQUEST, "namespace required")
}

async fn status_namespace(
    Path(namespace): Path<String>,
    State(state): State<AppState>,
) -> Response {
    let handle = match state.namespaces.get_or_create(&namespace).await {
        Ok(handle) => handle,
        Err(NamespaceError::NotAllowed) => {
            return json_error(StatusCode::NOT_FOUND, "namespace not found");
        }
        Err(err) => {
            error!("Namespace error: {}", err);
            return json_error(StatusCode::INTERNAL_SERVER_ERROR, "namespace error");
        }
    };

    match handle.compactor.get_status().await {
        Ok(status) => Json(status).into_response(),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "failed to get status"),
    }
}

async fn compact_namespace(
    Path(namespace): Path<String>,
    State(state): State<AppState>,
) -> Response {
    let handle = match state.namespaces.get_or_create(&namespace).await {
        Ok(handle) => handle,
        Err(NamespaceError::NotAllowed) => {
            return json_error(StatusCode::NOT_FOUND, "namespace not found");
        }
        Err(err) => {
            error!("Namespace error: {}", err);
            return json_error(StatusCode::INTERNAL_SERVER_ERROR, "namespace error");
        }
    };

    match handle.compactor.run().await {
        Ok(_) => Json(serde_json::json!({"status": "compaction completed"})).into_response(),
        Err(err) => {
            error!("Manual compaction error: {}", err);
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "compaction failed")
        }
    }
}

async fn upsert(
    Path(namespace): Path<String>,
    State(state): State<AppState>,
    Json(req): Json<UpsertRequest>,
) -> Response {
    let handle = match state.namespaces.get_or_create(&namespace).await {
        Ok(handle) => handle,
        Err(NamespaceError::NotAllowed) => {
            return json_error(StatusCode::NOT_FOUND, "namespace not found");
        }
        Err(err) => {
            error!("Namespace error: {}", err);
            return json_error(StatusCode::INTERNAL_SERVER_ERROR, "namespace error");
        }
    };

    if req.vectors.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "vectors are required");
    }

    // Validate each document has a non-empty vector
    for (i, doc) in req.vectors.iter().enumerate() {
        if doc.vector.is_empty() {
            return json_error(
                StatusCode::BAD_REQUEST,
                &format!("document at index {} is missing required 'vector' field", i),
            );
        }
    }

    // Write to WAL for durability (query nodes will scan WAL for real-time visibility)
    if let Err(err) = handle.wal_writer.write_upsert(req.vectors.clone()).await {
        error!("Upsert error: {}", err);
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, "upsert failed");
    }

    Json(UpsertResponse { status: "ok".to_string() }).into_response()
}

async fn delete_vectors(
    Path(namespace): Path<String>,
    State(state): State<AppState>,
    Json(req): Json<DeleteRequest>,
) -> Response {
    let handle = match state.namespaces.get_or_create(&namespace).await {
        Ok(handle) => handle,
        Err(NamespaceError::NotAllowed) => {
            return json_error(StatusCode::NOT_FOUND, "namespace not found");
        }
        Err(err) => {
            error!("Namespace error: {}", err);
            return json_error(StatusCode::INTERNAL_SERVER_ERROR, "namespace error");
        }
    };

    if req.ids.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "ids are required");
    }

    // Write to WAL for durability (query nodes will scan WAL for real-time visibility)
    if let Err(err) = handle.wal_writer.write_delete(req.ids.clone()).await {
        error!("Delete error: {}", err);
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, "delete failed");
    }

    Json(DeleteResponse { status: "ok".to_string() }).into_response()
}

fn json_error(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({"error": message});
    (status, Json(body)).into_response()
}
