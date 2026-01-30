# Single Dockerfile for all deployment targets
#
# Build Arguments:
#   SERVICE - which service to build: "query", "ingest", or "all" (default: all)
#
# Railway Template Usage (builds all, selects at runtime):
#   docker build -t tidepool .
#   - tidepool-query service:  Start Command = /tidepool-query
#   - tidepool-ingest service: Start Command = /tidepool-ingest
#
# GitHub Release Usage (builds single service per image):
#   docker build --build-arg SERVICE=query -t tidepool-query .
#   docker build --build-arg SERVICE=ingest -t tidepool-ingest .
#
# Local Usage:
#   docker build -t tidepool .
#   docker run -p 8080:8080 tidepool /tidepool-query
#   docker run -p 8080:8080 tidepool /tidepool-ingest

ARG SERVICE=all

# =============================================================================
# Builder stage - compiles services based on SERVICE arg
# =============================================================================
FROM rust:1.93-slim AS builder

ARG SERVICE

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifests for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/tidepool-common/Cargo.toml crates/tidepool-common/Cargo.toml
COPY query/Cargo.toml query/Cargo.toml
COPY ingest/Cargo.toml ingest/Cargo.toml
COPY perf/Cargo.toml perf/Cargo.toml

# Create dummy source files for dependency caching
RUN mkdir -p crates/tidepool-common/src query/src ingest/src perf/src \
    && echo "fn main() {}" > query/src/main.rs \
    && echo "fn main() {}" > ingest/src/main.rs \
    && echo "fn main() {}" > perf/src/main.rs \
    && echo "pub fn dummy() {}" > crates/tidepool-common/src/lib.rs

# Build dependencies only (this layer is cached)
RUN if [ "$SERVICE" = "query" ]; then \
        cargo build --release -p tidepool-query || true; \
    elif [ "$SERVICE" = "ingest" ]; then \
        cargo build --release -p tidepool-ingest || true; \
    else \
        cargo build --release -p tidepool-query -p tidepool-ingest || true; \
    fi

# Copy actual source code
COPY . .

# Touch source files to invalidate the cache for actual build
RUN touch crates/tidepool-common/src/lib.rs \
    && touch query/src/main.rs \
    && touch ingest/src/main.rs

# Build service(s) based on SERVICE arg
RUN if [ "$SERVICE" = "query" ]; then \
        cargo build --release -p tidepool-query; \
    elif [ "$SERVICE" = "ingest" ]; then \
        cargo build --release -p tidepool-ingest; \
    else \
        cargo build --release -p tidepool-query -p tidepool-ingest; \
    fi

# =============================================================================
# Runtime stage - minimal production image
# =============================================================================
FROM debian:trixie-slim AS runtime

ARG SERVICE

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create data directory for caching
RUN mkdir -p /data

# Copy service binaries based on SERVICE arg
# Using shell script to conditionally copy files
COPY --from=builder /build/target/release/tidepool-* /tmp/

RUN if [ "$SERVICE" = "query" ]; then \
        mv /tmp/tidepool-query /tidepool-query; \
    elif [ "$SERVICE" = "ingest" ]; then \
        mv /tmp/tidepool-ingest /tidepool-ingest; \
    else \
        mv /tmp/tidepool-query /tidepool-query; \
        mv /tmp/tidepool-ingest /tidepool-ingest; \
    fi && rm -rf /tmp/tidepool-*

# Persist SERVICE as environment variable for runtime CMD
ENV SERVICE=${SERVICE}

WORKDIR /

EXPOSE 8080

# Set default command based on SERVICE
# - SERVICE=query → runs query service
# - SERVICE=ingest → runs ingest service  
# - SERVICE=all → defaults to query (Railway overrides via Start Command)
CMD ["/bin/sh", "-c", "if [ \"$SERVICE\" = \"ingest\" ]; then exec /tidepool-ingest; else exec /tidepool-query; fi"]
