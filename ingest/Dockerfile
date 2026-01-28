FROM golang:1.23-alpine AS builder

# Install git for go mod download (some deps may need it)
RUN apk add --no-cache git ca-certificates

WORKDIR /build

# Copy go mod files first for better cache
COPY go.mod go.sum* ./
RUN go mod download

# Copy source code
COPY . .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s" \
    -o tidepool-ingest \
    ./ingest/cmd/ingest

# Final stage - distroless for minimal attack surface
FROM gcr.io/distroless/static:nonroot

# Copy the binary
COPY --from=builder /build/tidepool-ingest /tidepool-ingest

# Copy CA certificates for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Create data directory
WORKDIR /

# Run as non-root user
USER nonroot:nonroot

EXPOSE 8080

ENTRYPOINT ["/tidepool-ingest"]
