# Tidepool Makefile
# 
# Usage:
#   make build       - Build all services
#   make test        - Run all tests
#   make bench       - Run HTTP API benchmarks
#   make bench-local - Run benchmarks against local Docker stack
#   make docker-up   - Start local Docker stack
#   make docker-down - Stop local Docker stack

.PHONY: build test bench bench-local bench-rust docker-up docker-down clean fmt lint

# Default target
all: build test

# =============================================================================
# Build
# =============================================================================

build:
	cargo build --release

build-query:
	cargo build --release -p tidepool-query

build-ingest:
	cargo build --release -p tidepool-ingest

check:
	cargo check --all

# =============================================================================
# Test
# =============================================================================

test:
	cargo test --all

test-verbose:
	cargo test --all -- --nocapture

# =============================================================================
# Benchmarks
# =============================================================================

# Run HTTP API benchmarks against production (prompts for URLs if not provided)
bench:
	@./scripts/bench.sh "$(INGEST_URL)" "$(QUERY_URL)"

# Run HTTP API benchmarks against local Docker stack
bench-local: docker-up
	@echo "Waiting for services to be ready..."
	@sleep 3
	@./scripts/bench.sh "http://localhost:8081" "http://localhost:8080"

# Run Rust-level benchmarks (HNSW, SIMD, etc.)
bench-rust:
	cargo bench -p tidepool-common

# =============================================================================
# Docker
# =============================================================================

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose up --build -d

docker-clean:
	docker-compose down -v

# =============================================================================
# Code Quality
# =============================================================================

fmt:
	cargo fmt --all

lint:
	cargo clippy --all -- -D warnings

# =============================================================================
# Clean
# =============================================================================

clean:
	cargo clean
