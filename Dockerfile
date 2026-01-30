# Single Dockerfile for Railway template deployment
# Builds ALL services into one image - select which to run via start command
#
# Railway Template Usage:
#   - tidepool-query service:  Start Command = /tidepool-query
#   - tidepool-ingest service: Start Command = /tidepool-ingest
#
# Local Usage:
#   docker build -t tidepool .
#   docker run -p 8080:8080 tidepool /tidepool-query
#   docker run -p 8080:8080 tidepool /tidepool-ingest

# =============================================================================
# Builder stage - compiles all services
# =============================================================================
FROM rust:1.93-slim AS builder

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
RUN cargo build --release -p tidepool-query -p tidepool-ingest || true

# Copy actual source code
COPY . .

# Touch source files to invalidate the cache for actual build
RUN touch crates/tidepool-common/src/lib.rs \
    && touch query/src/main.rs \
    && touch ingest/src/main.rs

# Build all services
RUN cargo build --release -p tidepool-query -p tidepool-ingest

# =============================================================================
# Runtime stage - minimal production image with all service binaries
# =============================================================================
FROM debian:trixie-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create data directory for caching
RUN mkdir -p /data

# Copy ALL service binaries
COPY --from=builder /build/target/release/tidepool-query /tidepool-query
COPY --from=builder /build/target/release/tidepool-ingest /tidepool-ingest

WORKDIR /

EXPOSE 8080

# Default to query service (can be overridden via Railway Start Command)
CMD ["/tidepool-query"]
