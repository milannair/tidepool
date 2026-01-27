# Default Dockerfile for environments that expect a root Dockerfile.
# For Railway templates, prefer the service-specific Dockerfiles in services/*.

FROM golang:1.23-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /build

COPY go.mod go.sum* ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s" \
    -o tidepool-query \
    ./cmd/tidepool-query

FROM gcr.io/distroless/static:nonroot

COPY --from=builder /build/tidepool-query /tidepool-query
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

WORKDIR /
USER nonroot:nonroot
EXPOSE 8080

ENTRYPOINT ["/tidepool-query"]
