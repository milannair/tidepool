#!/bin/bash
#
# Tidepool HTTP API Benchmark Script
#
# Usage:
#   ./scripts/bench.sh <INGEST_URL> <QUERY_URL>
#   ./scripts/bench.sh  # Interactive mode - prompts for URLs
#
# Example:
#   ./scripts/bench.sh http://localhost:8081 http://localhost:8080
#   ./scripts/bench.sh https://ingest.railway.app https://query.railway.app

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

INGEST_URL="$1"
QUERY_URL="$2"

# Prompt for URLs if not provided
if [ -z "$INGEST_URL" ] || [ "$INGEST_URL" = " " ]; then
    echo -e "${BLUE}Tidepool Benchmark Setup${NC}"
    echo ""
    echo -e "Enter your service URLs (or press Enter for localhost defaults):"
    echo ""
    read -p "Ingest URL [http://localhost:8081]: " INGEST_URL
    INGEST_URL="${INGEST_URL:-http://localhost:8081}"
fi

if [ -z "$QUERY_URL" ] || [ "$QUERY_URL" = " " ]; then
    read -p "Query URL [http://localhost:8080]: " QUERY_URL
    QUERY_URL="${QUERY_URL:-http://localhost:8080}"
    echo ""
fi

NAMESPACE="bench-$(date +%s)"
DIM=128

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Tidepool Performance Benchmark                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Ingest URL: ${GREEN}$INGEST_URL${NC}"
echo -e "Query URL:  ${GREEN}$QUERY_URL${NC}"
echo -e "Namespace:  ${GREEN}$NAMESPACE${NC}"
echo -e "Dimensions: ${GREEN}$DIM${NC}"
echo ""

# Check services are healthy
echo -e "${YELLOW}[1/5] Checking service health...${NC}"
ingest_health=$(curl -s "$INGEST_URL/health" 2>/dev/null || echo "failed")
query_health=$(curl -s "$QUERY_URL/health" 2>/dev/null || echo "failed")

if [[ "$ingest_health" != *"healthy"* ]]; then
    echo -e "${RED}✗ Ingest service not healthy: $ingest_health${NC}"
    exit 1
fi
if [[ "$query_health" != *"healthy"* ]]; then
    echo -e "${RED}✗ Query service not healthy: $query_health${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Both services healthy${NC}"
echo ""

# Generate vectors helper
generate_batch() {
    local start=$1
    local count=$2
    python3 -c "
import json
vectors = []
for i in range($start, $start + $count):
    v = [0.0] * $DIM
    v[i % $DIM] = 1.0
    v[(i + 1) % $DIM] = 0.5
    vectors.append({'id': f'v{i}', 'vector': v})
print(json.dumps({'vectors': vectors}))
"
}

# Generate query vector
QUERY_VEC=$(python3 -c "import json; v=[0.0]*$DIM; v[0]=1.0; print(json.dumps(v))")
QUERY_PAYLOAD="{\"vector\":$QUERY_VEC,\"top_k\":10}"

# =============================================================================
# Test 1: Insert Throughput (Sequential)
# =============================================================================
echo -e "${YELLOW}[2/5] Testing insert throughput (sequential)...${NC}"

start_time=$(python3 -c "import time; print(time.time())")
for i in $(seq 0 9); do
    generate_batch $((i * 100)) 100 | \
        curl -s -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
        -H "Content-Type: application/json" -d @- > /dev/null
done
end_time=$(python3 -c "import time; print(time.time())")
duration=$(python3 -c "print(round($end_time - $start_time, 2))")
seq_rate=$(python3 -c "print(round(1000 / $duration, 2))")

echo -e "  Sequential: 1,000 vectors in ${duration}s = ${GREEN}$seq_rate vec/s${NC}"

# =============================================================================
# Test 2: Insert Throughput (Parallel)
# =============================================================================
echo -e "${YELLOW}[3/5] Testing insert throughput (parallel)...${NC}"

NAMESPACE2="${NAMESPACE}-parallel"
start_time=$(python3 -c "import time; print(time.time())")
for i in $(seq 0 49); do
    generate_batch $((i * 100)) 100 | \
        curl -s -X POST "$INGEST_URL/v1/vectors/$NAMESPACE2" \
        -H "Content-Type: application/json" -d @- > /dev/null &
    # Limit to 10 concurrent
    if (( (i + 1) % 10 == 0 )); then wait; fi
done
wait
end_time=$(python3 -c "import time; print(time.time())")
duration=$(python3 -c "print(round($end_time - $start_time, 2))")
par_rate=$(python3 -c "print(round(5000 / $duration, 2))")

echo -e "  Parallel:   5,000 vectors in ${duration}s = ${GREEN}$par_rate vec/s${NC}"
echo ""

# =============================================================================
# Test 3: Query Throughput (Sequential)
# =============================================================================
echo -e "${YELLOW}[4/5] Testing query throughput (sequential)...${NC}"

start_time=$(python3 -c "import time; print(time.time())")
for i in $(seq 1 20); do
    curl -s -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
        -H "Content-Type: application/json" \
        -d "$QUERY_PAYLOAD" > /dev/null
done
end_time=$(python3 -c "import time; print(time.time())")
duration=$(python3 -c "print(round($end_time - $start_time, 2))")
seq_qps=$(python3 -c "print(round(20 / $duration, 2))")
seq_latency=$(python3 -c "print(round($duration * 1000 / 20, 2))")

echo -e "  Sequential: 20 queries in ${duration}s = ${GREEN}$seq_qps QPS${NC} (${seq_latency}ms avg)"

# =============================================================================
# Test 4: Query Throughput (Parallel)
# =============================================================================
echo -e "${YELLOW}[5/5] Testing query throughput (parallel)...${NC}"

# Test different concurrency levels
for conc in 50 100 200; do
    start_time=$(python3 -c "import time; print(time.time())")
    for i in $(seq 1 $conc); do
        curl -s -X POST "$QUERY_URL/v1/vectors/$NAMESPACE2" \
            -H "Content-Type: application/json" \
            -d "$QUERY_PAYLOAD" > /dev/null &
    done
    wait
    end_time=$(python3 -c "import time; print(time.time())")
    duration=$(python3 -c "print(round($end_time - $start_time, 2))")
    qps=$(python3 -c "print(round($conc / $duration, 2))")
    echo -e "  Concurrency $conc: ${conc} queries in ${duration}s = ${GREEN}$qps QPS${NC}"
done
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                        Summary                               ║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║${NC}  Insert (sequential):  ${GREEN}$seq_rate vec/s${NC}"
echo -e "${BLUE}║${NC}  Insert (parallel):    ${GREEN}$par_rate vec/s${NC}"
echo -e "${BLUE}║${NC}  Query (sequential):   ${GREEN}$seq_qps QPS${NC} @ ${seq_latency}ms"
echo -e "${BLUE}║${NC}  Query (parallel):     See above for different concurrency levels"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Full report: ${YELLOW}docs/PERFORMANCE.md${NC}"
