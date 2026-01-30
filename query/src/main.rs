use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tower_http::cors::{Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tracing::{error, info};

use tidepool_common::config::Config;
use tidepool_common::document::QueryRequest;
use tidepool_common::storage::{LocalStore, S3Store};
use tidepool_query::bootstrap::{Bootstrap, Rehydrator};
use tidepool_query::engine::EngineOptions;
use tidepool_query::eviction::Evictor;
use tidepool_query::loader::DataLoader;
use tidepool_query::namespace_manager::{NamespaceError, NamespaceManager};
use tidepool_query::sync::BackgroundSync;

#[derive(Clone)]
struct AppState {
    namespaces: Arc<NamespaceManager<LocalStore>>,
    max_top_k: usize,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Starting tidepool-query service (local-first)...");

    let cfg = match Config::from_env() {
        Ok(cfg) => cfg,
        Err(err) => {
            error!("Configuration error: {}", err);
            std::process::exit(1);
        }
    };

    let s3_store = match S3Store::new(&cfg).await {
        Ok(store) => store,
        Err(err) => {
            error!("Failed to initialize S3 client: {}", err);
            std::process::exit(1);
        }
    };

    // Two-phase startup: Phase 1 loads manifests + Bloom filters only (fast)
    // Phase 2 rehydrates segment data in background
    if cfg.eager_sync_all {
        // Legacy mode: download everything at startup (blocking)
        info!("EAGER_SYNC_ALL=true: downloading all data at startup...");
        let loader = DataLoader::new(s3_store.clone(), PathBuf::from(&cfg.data_dir));
        match loader.sync_all().await {
            Ok(stats) => info!(
                "Synced {} bytes ({} namespaces, {} segments) in {:?}",
                stats.bytes_synced,
                stats.namespaces,
                stats.segments,
                stats.duration
            ),
            Err(err) => {
                error!("Initial sync failed: {}", err);
                std::process::exit(1);
            }
        }
    } else {
        // Cold-start-safe mode: two-phase bootstrap
        info!("Phase 1: Loading manifests and Bloom filters...");
        let bootstrap = Bootstrap::new(
            s3_store.clone(),
            PathBuf::from(&cfg.data_dir),
            cfg.max_local_disk,
            cfg.target_local_disk,
            cfg.eager_sync_all,
        );
        
        let bootstrap_state = match bootstrap.phase1().await {
            Ok(state) => state,
            Err(err) => {
                error!("Phase 1 bootstrap failed: {}", err);
                std::process::exit(1);
            }
        };

        info!(
            "Phase 1 complete: {} namespaces, {} Bloom filters loaded, {} segments to download",
            bootstrap_state.stats.namespaces,
            bootstrap_state.stats.blooms_loaded,
            bootstrap_state.to_download.len()
        );

        // Start Phase 2 (background rehydration) if there are segments to download
        if !bootstrap_state.to_download.is_empty() {
            info!("Phase 2: Starting background rehydration ({} segments)...", bootstrap_state.to_download.len());
            let rehydrator = Rehydrator::new(
                s3_store.clone(),
                PathBuf::from(&cfg.data_dir),
                cfg.max_local_disk,
                bootstrap_state.to_download,
            );
            tokio::spawn(async move {
                rehydrator.run().await;
            });
        }
    }

    let local_store = LocalStore::new(&cfg.data_dir);

    if cfg.hot_buffer_max_size > 0 {
        info!(
            "Hot buffer enabled (buffer size per namespace: {})",
            cfg.hot_buffer_max_size
        );
    } else {
        info!("Hot buffer disabled (HOT_BUFFER_MAX_SIZE=0)");
    }

    let namespaces = NamespaceManager::new(
        local_store,
        Some(cfg.data_dir.clone()),
        EngineOptions {
            hnsw_ef_search: cfg.hnsw_ef_search,
            quantization_rerank_factor: cfg.quantization_rerank_factor,
            bm25_k1: cfg.bm25_k1,
            bm25_b: cfg.bm25_b,
            rrf_k: cfg.rrf_k,
            tokenizer_config: cfg.tokenizer_config(),
        },
        cfg.hot_buffer_max_size,
        cfg.allowed_namespaces.clone(),
        cfg.max_namespaces,
        cfg.namespace_idle_timeout,
        cfg.namespace.clone(),
    );
    let namespaces = Arc::new(namespaces);

    let sync = BackgroundSync::new(
        s3_store,
        PathBuf::from(&cfg.data_dir),
        namespaces.clone(),
        cfg.sync_interval,
    )
    .with_eager_sync(cfg.eager_sync_all);
    tokio::spawn(sync.run());

    // Start background eviction (checks every 60 seconds)
    let evictor = Evictor::new(
        PathBuf::from(&cfg.data_dir),
        cfg.max_local_disk,
        cfg.target_local_disk,
    );
    tokio::spawn(evictor.run(Duration::from_secs(60)));

    let state = AppState {
        namespaces,
        max_top_k: cfg.max_top_k,
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
        .route("/", get(root))
        .route("/health", get(health))
        .route("/v1/namespaces", get(list_namespaces))
        .route("/v1/namespaces/", get(list_namespaces))
        .route("/v1/namespaces/:namespace", get(get_namespace))
        .route("/v1/namespaces/:namespace/query", post(query))
        .route("/v1/vectors/:namespace", post(query))
        .layer(cors)
        .layer(RequestBodyLimitLayer::new(cfg.max_body_bytes))
        .with_state(state);

    let addr: SocketAddr = format!("0.0.0.0:{}", cfg.port)
        .parse()
        .expect("invalid port");

    info!("Listening on {}", addr);
    axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app)
        .await
        .unwrap();
}

async fn root() -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "tidepool",
        "version": "0.1.0"
    }))
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "tidepool-query"
    }))
}

async fn list_namespaces(State(state): State<AppState>) -> impl IntoResponse {
    match state.namespaces.list_namespaces().await {
        Ok(namespaces) => Json(serde_json::json!({"namespaces": namespaces})).into_response(),
        Err(err) => {
            error!("List namespaces error: {}", err);
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "failed to list namespaces")
        }
    }
}

async fn get_namespace(Path(namespace): Path<String>, State(state): State<AppState>) -> Response {
    match state.namespaces.get_namespace_info(&namespace).await {
        Ok(info) => Json(info).into_response(),
        Err(NamespaceError::NotAllowed) => json_error(StatusCode::NOT_FOUND, "namespace not found"),
        Err(err) => {
            error!("Get namespace error: {}", err);
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "failed to get namespace")
        }
    }
}

async fn query(
    Path(namespace): Path<String>,
    State(state): State<AppState>,
    Json(mut req): Json<QueryRequest>,
) -> Response {
    let engine = match state.namespaces.get_engine(&namespace).await {
        Ok(engine) => engine,
        Err(NamespaceError::NotAllowed) => {
            return json_error(StatusCode::NOT_FOUND, "namespace not found");
        }
        Err(err) => {
            error!("Namespace error: {}", err);
            return json_error(StatusCode::INTERNAL_SERVER_ERROR, "namespace error");
        }
    };

    let has_vector = !req.vector.is_empty();
    let has_text = req
        .text
        .as_ref()
        .map(|t| !t.trim().is_empty())
        .unwrap_or(false);

    let mode = match req.mode.as_deref() {
        Some("vector") => "vector",
        Some("text") => "text",
        Some("hybrid") => "hybrid",
        None => {
            if has_vector && has_text {
                "hybrid"
            } else if has_text {
                "text"
            } else {
                "vector"
            }
        }
        _ => "vector",
    };

    if mode == "vector" && !has_vector {
        return json_error(StatusCode::BAD_REQUEST, "vector is required");
    }
    if mode == "text" && !has_text {
        return json_error(StatusCode::BAD_REQUEST, "text is required");
    }
    if mode == "hybrid" && (!has_vector || !has_text) {
        return json_error(StatusCode::BAD_REQUEST, "vector and text are required for hybrid");
    }

    if req.top_k == 0 {
        req.top_k = 10;
    }
    if state.max_top_k > 0 && req.top_k > state.max_top_k {
        req.top_k = state.max_top_k;
    }

    if req.ef_search == 0 {
        req.ef_search = 0;
    }

    let stats = engine.get_stats().await;
    let uses_vector = mode != "text";
    if uses_vector && stats.dimensions > 0 && req.vector.len() != stats.dimensions {
        return json_error(StatusCode::BAD_REQUEST, "vector dimensions do not match namespace");
    }

    match engine.query(&req).await {
        Ok(resp) => Json(resp).into_response(),
        Err(err) => {
            error!("Query error: {}", err);
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "query failed")
        }
    }
}

fn json_error(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({"error": message});
    (status, Json(body)).into_response()
}
