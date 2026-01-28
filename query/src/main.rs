use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tower_http::cors::{Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tracing::{error, info};

use tidepool_common::config::Config;
use tidepool_common::document::{NamespaceInfo, QueryRequest};
use tidepool_common::storage::S3Store;
use tidepool_query::engine::{Engine, EngineOptions};

#[derive(Clone)]
struct AppState {
    engine: Arc<Engine<S3Store>>,
    namespace: String,
    max_top_k: usize,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Starting tidepool-query service...");

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

    let engine = Engine::new_with_options(
        storage,
        cfg.namespace.clone(),
        Some(cfg.cache_dir.clone()),
        EngineOptions {
            hnsw_ef_search: cfg.hnsw_ef_search,
        },
    );

    if let Err(err) = engine.load_manifest().await {
        info!("Warning: failed to load initial manifest: {}", err);
    }

    let state = AppState {
        engine: Arc::new(engine),
        namespace: cfg.namespace.clone(),
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
    let stats = state.engine.get_stats().await;
    let namespaces = vec![NamespaceInfo {
        namespace: state.namespace.clone(),
        approx_count: stats.total_vectors,
        dimensions: stats.dimensions,
    }];
    Json(serde_json::json!({"namespaces": namespaces}))
}

async fn get_namespace(Path(namespace): Path<String>, State(state): State<AppState>) -> Response {
    if namespace != state.namespace {
        return json_error(StatusCode::NOT_FOUND, "namespace not found");
    }
    let stats = state.engine.get_stats().await;
    let info = NamespaceInfo {
        namespace: state.namespace.clone(),
        approx_count: stats.total_vectors,
        dimensions: stats.dimensions,
    };
    Json(info).into_response()
}

async fn query(
    Path(namespace): Path<String>,
    State(state): State<AppState>,
    Json(mut req): Json<QueryRequest>,
) -> Response {
    if namespace != state.namespace {
        return json_error(StatusCode::NOT_FOUND, "namespace not found");
    }

    if req.vector.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "vector is required");
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

    let stats = state.engine.get_stats().await;
    if stats.dimensions > 0 && req.vector.len() != stats.dimensions {
        return json_error(StatusCode::BAD_REQUEST, "vector dimensions do not match namespace");
    }

    match state.engine.query(&req).await {
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
