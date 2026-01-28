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
    -o tidepool-query \
    ./query/cmd/query

# Final stage - minimal alpine for volume support
FROM alpine:3.20

# Install CA certificates
RUN apk add --no-cache ca-certificates

# Create data directory
RUN mkdir -p /data

# Copy the binary
COPY --from=builder /build/tidepool-query /tidepool-query

WORKDIR /

EXPOSE 8080

ENTRYPOINT ["/tidepool-query"]
