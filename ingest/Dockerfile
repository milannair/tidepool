FROM rust:1.93-slim AS builder

WORKDIR /build

# Cache dependencies
COPY Cargo.toml Cargo.lock* ./
COPY crates/tidepool-common/Cargo.toml crates/tidepool-common/Cargo.toml
COPY query/Cargo.toml query/Cargo.toml
COPY ingest/Cargo.toml ingest/Cargo.toml

# Copy source
COPY . .

# Build ingest binary
RUN cargo build --release -p tidepool-ingest

FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/tidepool-ingest /tidepool-ingest

WORKDIR /

EXPOSE 8080

ENTRYPOINT ["/tidepool-ingest"]
