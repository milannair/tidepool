FROM rust:1.93-slim AS builder

WORKDIR /build

# Cache dependencies
COPY Cargo.toml Cargo.lock* ./
COPY crates/tidepool-common/Cargo.toml crates/tidepool-common/Cargo.toml
COPY query/Cargo.toml query/Cargo.toml
COPY ingest/Cargo.toml ingest/Cargo.toml

# Copy source
COPY . .

# Build query binary
RUN cargo build --release -p tidepool-query

FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data

COPY --from=builder /build/target/release/tidepool-query /tidepool-query

WORKDIR /

EXPOSE 8080

ENTRYPOINT ["/tidepool-query"]
