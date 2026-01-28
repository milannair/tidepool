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
use tidepool_common::wal::Writer as WalWriter;

mod compactor;
use compactor::Compactor;
mod wal_buffer;
use wal_buffer::BufferedWalWriter;

#[derive(Clone)]
struct AppState {
    wal_writer: Arc<BufferedWalWriter<S3Store>>,
    compactor: Arc<Compactor<S3Store>>,
    namespace: String,
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

    let wal_writer = BufferedWalWriter::new(
        WalWriter::new(storage.clone(), cfg.namespace.clone()),
        cfg.wal_batch_max_entries,
        cfg.wal_batch_flush_interval,
    );
    let compactor = Compactor::new_with_options(
        storage,
        cfg.namespace.clone(),
        WriterOptions {
            hnsw_m: cfg.hnsw_m,
            hnsw_ef_construction: cfg.hnsw_ef_construction,
            hnsw_ef_search: cfg.hnsw_ef_search,
            metric: tidepool_common::vector::DistanceMetric::Cosine,
        },
    );

    let compactor_task = {
        let compactor = compactor.clone();
        let interval = cfg.compaction_interval;
        tokio::spawn(async move {
            compactor.run_periodically(interval).await;
        })
    };

    let state = AppState {
        wal_writer: Arc::new(wal_writer),
        compactor: Arc::new(compactor),
        namespace: cfg.namespace.clone(),
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

    compactor_task.abort();
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "tidepool-ingest"
    }))
}

async fn status(State(state): State<AppState>) -> Response {
    match state.compactor.get_status().await {
        Ok(status) => Json(status).into_response(),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "failed to get status"),
    }
}

async fn compact(State(state): State<AppState>) -> Response {
    match state.compactor.run().await {
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
    if namespace != state.namespace {
        return json_error(StatusCode::NOT_FOUND, "namespace not found");
    }

    if req.vectors.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "vectors are required");
    }

    match state.wal_writer.write_upsert(req.vectors).await {
        Ok(_) => Json(UpsertResponse { status: "ok".to_string() }).into_response(),
        Err(err) => {
            error!("Upsert error: {}", err);
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "upsert failed")
        }
    }
}

async fn delete_vectors(
    Path(namespace): Path<String>,
    State(state): State<AppState>,
    Json(req): Json<DeleteRequest>,
) -> Response {
    if namespace != state.namespace {
        return json_error(StatusCode::NOT_FOUND, "namespace not found");
    }

    if req.ids.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "ids are required");
    }

    match state.wal_writer.write_delete(req.ids).await {
        Ok(_) => Json(DeleteResponse { status: "ok".to_string() }).into_response(),
        Err(err) => {
            error!("Delete error: {}", err);
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "delete failed")
        }
    }
}

fn json_error(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({"error": message});
    (status, Json(body)).into_response()
}
