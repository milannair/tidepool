use std::env;
use std::time::Duration;

use humantime::parse_duration;

use crate::quantization::QuantizationKind;
use crate::text::TokenizerConfig;

#[derive(Debug, Clone)]
pub struct Config {
    pub aws_access_key_id: String,
    pub aws_secret_access_key: String,
    pub aws_endpoint_url: String,
    pub aws_region: String,
    pub bucket_name: String,
    pub cache_dir: String,
    pub namespace: Option<String>,
    pub allowed_namespaces: Option<Vec<String>>,
    pub max_namespaces: Option<usize>,
    pub namespace_idle_timeout: Option<Duration>,
    pub compaction_interval: Duration,
    pub port: String,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    pub idle_timeout: Duration,
    pub max_body_bytes: usize,
    pub max_top_k: usize,
    pub cors_allow_origin: String,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub ivf_enabled: bool,
    pub ivf_min_segment_size: usize,
    pub ivf_nprobe_default: usize,
    pub ivf_k_factor: f32,
    pub ivf_min_k: usize,
    pub ivf_max_k: usize,
    pub quantization: QuantizationKind,
    pub quantization_rerank_factor: usize,
    pub wal_batch_max_entries: usize,
    pub wal_batch_flush_interval: Duration,
    // Real-time updates (WAL-based)
    pub hot_buffer_max_size: usize,
    /// Minimum interval between S3 state refreshes in milliseconds (default: 200)
    /// Lower = more real-time, higher = fewer S3 calls
    pub refresh_interval_ms: u64,
    /// How often to re-list WAL files from S3 in milliseconds (default: 5000)
    pub wal_list_interval_ms: u64,
    // Full-text search
    pub text_index_enabled: bool,
    pub bm25_k1: f32,
    pub bm25_b: f32,
    pub rrf_k: usize,
    pub text_enable_stemming: bool,
    pub text_language: String,
    pub text_stopwords: Option<Vec<String>>,
    pub text_min_token_len: usize,
    pub text_max_token_len: usize,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let cfg = Self {
            aws_access_key_id: env::var("AWS_ACCESS_KEY_ID").unwrap_or_default(),
            aws_secret_access_key: env::var("AWS_SECRET_ACCESS_KEY").unwrap_or_default(),
            aws_endpoint_url: env::var("AWS_ENDPOINT_URL").unwrap_or_default(),
            aws_region: get_env_with_fallback("AWS_DEFAULT_REGION", "AWS_REGION"),
            bucket_name: get_env_with_fallback("AWS_S3_BUCKET_NAME", "BUCKET_NAME"),
            cache_dir: env::var("CACHE_DIR").unwrap_or_else(|_| "/data".to_string()),
            namespace: parse_optional_string("NAMESPACE"),
            allowed_namespaces: parse_optional_csv("ALLOWED_NAMESPACES"),
            max_namespaces: parse_optional_usize("MAX_NAMESPACES"),
            namespace_idle_timeout: parse_optional_duration("NAMESPACE_IDLE_TIMEOUT"),
            compaction_interval: parse_duration_fallback("COMPACTION_INTERVAL", Duration::from_secs(300)),
            port: env::var("PORT").unwrap_or_else(|_| "8080".to_string()),
            read_timeout: parse_duration_fallback("READ_TIMEOUT", Duration::from_secs(30)),
            write_timeout: parse_duration_fallback("WRITE_TIMEOUT", Duration::from_secs(60)),
            idle_timeout: parse_duration_fallback("IDLE_TIMEOUT", Duration::from_secs(60)),
            max_body_bytes: parse_usize("MAX_BODY_BYTES", 25 * 1024 * 1024),
            max_top_k: parse_usize("MAX_TOP_K", 1000),
            cors_allow_origin: env::var("CORS_ALLOW_ORIGIN").unwrap_or_else(|_| "*".to_string()),
            hnsw_m: parse_usize("HNSW_M", 16),
            hnsw_ef_construction: parse_usize("HNSW_EF_CONSTRUCTION", 200),
            hnsw_ef_search: parse_usize("HNSW_EF_SEARCH", 100),
            ivf_enabled: parse_bool("IVF_ENABLED", true),
            ivf_min_segment_size: parse_usize("IVF_MIN_SEGMENT_SIZE", 10_000),
            ivf_nprobe_default: parse_usize("IVF_NPROBE_DEFAULT", 10),
            ivf_k_factor: parse_f32("IVF_K_FACTOR", 1.0),
            ivf_min_k: parse_usize("IVF_MIN_K", 16),
            ivf_max_k: parse_usize("IVF_MAX_K", 65_535),
            quantization: QuantizationKind::parse(
                env::var("QUANTIZATION").ok().as_deref().or(Some("sq8"))
            ),
            quantization_rerank_factor: parse_usize("QUANTIZATION_RERANK_FACTOR", 4),
            wal_batch_max_entries: parse_usize("WAL_BATCH_MAX_ENTRIES", 1),
            wal_batch_flush_interval: parse_duration_fallback("WAL_BATCH_FLUSH_INTERVAL", Duration::from_millis(0)),
            // Real-time updates (WAL-based)
            hot_buffer_max_size: parse_usize("HOT_BUFFER_MAX_SIZE", 10_000),
            refresh_interval_ms: parse_usize("REFRESH_INTERVAL_MS", 200) as u64,
            wal_list_interval_ms: parse_usize("WAL_LIST_INTERVAL_MS", 5000) as u64,
            // Full-text search
            text_index_enabled: parse_bool("TEXT_INDEX_ENABLED", true),
            bm25_k1: parse_f32("BM25_K1", 1.2),
            bm25_b: parse_f32("BM25_B", 0.75),
            rrf_k: parse_usize("RRF_K", 60),
            text_enable_stemming: parse_bool("TEXT_ENABLE_STEMMING", true),
            text_language: env::var("TEXT_LANGUAGE").unwrap_or_else(|_| "english".to_string()),
            text_stopwords: parse_optional_csv("TEXT_STOPWORDS"),
            text_min_token_len: parse_usize("TEXT_MIN_TOKEN_LEN", 2),
            text_max_token_len: parse_usize("TEXT_MAX_TOKEN_LEN", 32),
        };

        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.aws_access_key_id.is_empty() {
            return Err(ConfigError::Missing("AWS_ACCESS_KEY_ID"));
        }
        if self.aws_secret_access_key.is_empty() {
            return Err(ConfigError::Missing("AWS_SECRET_ACCESS_KEY"));
        }
        if self.aws_endpoint_url.is_empty() {
            return Err(ConfigError::Missing("AWS_ENDPOINT_URL"));
        }
        if self.aws_region.is_empty() {
            return Err(ConfigError::Missing("AWS_DEFAULT_REGION or AWS_REGION"));
        }
        if self.bucket_name.is_empty() {
            return Err(ConfigError::Missing("AWS_S3_BUCKET_NAME or BUCKET_NAME"));
        }
        Ok(())
    }

    pub fn tokenizer_config(&self) -> TokenizerConfig {
        let mut cfg = TokenizerConfig::default()
            .with_language(self.text_language.clone());
        cfg.enable_stemming = self.text_enable_stemming;
        cfg.min_token_len = self.text_min_token_len;
        cfg.max_token_len = self.text_max_token_len;
        if let Some(list) = &self.text_stopwords {
            let set = list.iter().map(|s| s.to_lowercase()).collect();
            cfg = cfg.with_stopwords(set);
        }
        cfg
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("missing required configuration: {0}")]
    Missing(&'static str),
}

fn get_env_with_fallback(primary: &str, fallback: &str) -> String {
    env::var(primary).unwrap_or_else(|_| env::var(fallback).unwrap_or_default())
}

fn parse_optional_string(key: &str) -> Option<String> {
    match env::var(key) {
        Ok(raw) if !raw.trim().is_empty() => Some(raw),
        _ => None,
    }
}

fn parse_optional_csv(key: &str) -> Option<Vec<String>> {
    let raw = parse_optional_string(key)?;
    let values: Vec<String> = raw
        .split(',')
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .map(|v| v.to_string())
        .collect();
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

fn parse_optional_usize(key: &str) -> Option<usize> {
    let raw = parse_optional_string(key)?;
    let value = raw.parse::<usize>().ok()?;
    if value == 0 {
        None
    } else {
        Some(value)
    }
}

fn parse_optional_duration(key: &str) -> Option<Duration> {
    let raw = parse_optional_string(key)?;
    parse_duration(&raw).ok()
}

fn parse_duration_fallback(key: &str, default: Duration) -> Duration {
    match env::var(key) {
        Ok(raw) if !raw.is_empty() => parse_duration(&raw).unwrap_or(default),
        _ => default,
    }
}

fn parse_usize(key: &str, default: usize) -> usize {
    match env::var(key) {
        Ok(raw) if !raw.is_empty() => raw.parse::<usize>().unwrap_or(default),
        _ => default,
    }
}

fn parse_f32(key: &str, default: f32) -> f32 {
    match env::var(key) {
        Ok(raw) if !raw.is_empty() => raw.parse::<f32>().unwrap_or(default),
        _ => default,
    }
}

fn parse_bool(key: &str, default: bool) -> bool {
    match env::var(key) {
        Ok(raw) if !raw.is_empty() => match raw.to_lowercase().as_str() {
            "1" | "true" | "yes" | "y" | "on" => true,
            "0" | "false" | "no" | "n" | "off" => false,
            _ => default,
        },
        _ => default,
    }
}
