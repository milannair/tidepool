#!/bin/bash
#
# Tidepool Scaling Benchmark
#
# Tests query performance at increasing data sizes to find scaling limits.
# Empties the bucket between each test for clean measurements.
#
# Usage:
#   ./scripts/scale-bench.sh <INGEST_URL> <QUERY_URL>
#
# Example:
#   ./scripts/scale-bench.sh https://tidepool-ingest-production-973b.up.railway.app https://tidepool-query-production-a8f6.up.railway.app
#
# Requires:
#   - .env file with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AWS_REGION, BUCKET_NAME
#   - AWS CLI installed (aws s3 commands)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"

if [[ -f "$ENV_FILE" ]]; then
    echo -e "${YELLOW}Loading environment from $ENV_FILE${NC}"
    set -a
    source "$ENV_FILE"
    set +a
else
    echo -e "${RED}Warning: .env file not found at $ENV_FILE${NC}"
    echo -e "${RED}Bucket clearing will not work without S3 credentials.${NC}"
fi

INGEST_URL="${1:-http://localhost:8081}"
QUERY_URL="${2:-http://localhost:8080}"

# Test parameters
DIM=512  # Fixed test dimension
BATCH_SIZE=500
QUERY_SAMPLES=20
NAMESPACE="default"  # Use default namespace so we can clear it

# Scaling test points
SCALES=(1000 5000 10000 25000 50000 100000)

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Tidepool Scaling Benchmark                          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Ingest URL:  ${GREEN}$INGEST_URL${NC}"
echo -e "Query URL:   ${GREEN}$QUERY_URL${NC}"
echo -e "Namespace:   ${GREEN}$NAMESPACE${NC}"
echo -e "Dimensions:  ${GREEN}$DIM${NC}"
echo -e "Test scales: ${GREEN}${SCALES[*]}${NC}"
echo ""

# Check services
echo -e "${YELLOW}Checking service health...${NC}"
ingest_health=$(curl -s --max-time 10 "$INGEST_URL/health" 2>/dev/null || echo "failed")
query_health=$(curl -s --max-time 10 "$QUERY_URL/health" 2>/dev/null || echo "failed")

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

# Clear the S3 bucket completely (matches GitHub workflow)
clear_bucket() {
    echo -e "  ${YELLOW}Clearing bucket...${NC}"
    
    if [[ -z "$AWS_ACCESS_KEY_ID" ]] || [[ -z "$AWS_SECRET_ACCESS_KEY" ]] || [[ -z "$BUCKET_NAME" ]]; then
        echo -e "    ${RED}skipped (missing S3 credentials)${NC}"
        return 1
    fi
    
    local endpoint_flag=""
    local region_flag="--region us-east-1"
    if [[ -n "$AWS_ENDPOINT_URL" ]]; then
        endpoint_flag="--endpoint-url $AWS_ENDPOINT_URL"
    fi
    if [[ -n "$AWS_REGION" ]]; then
        region_flag="--region $AWS_REGION"
    fi
    
    # Delete all current objects
    echo -ne "    Deleting objects... "
    aws s3 rm "s3://$BUCKET_NAME" --recursive $endpoint_flag $region_flag 2>/dev/null || true
    echo -e "${GREEN}done${NC}"
    
    # Delete all object versions (if versioning enabled)
    echo -ne "    Deleting object versions... "
    aws s3api list-object-versions --bucket "$BUCKET_NAME" $endpoint_flag $region_flag \
        --query 'Versions[].{Key:Key,VersionId:VersionId}' --output json 2>/dev/null | \
        jq -r '.[] | "\(.Key) \(.VersionId)"' 2>/dev/null | \
        while read key version; do
            [ -n "$key" ] && aws s3api delete-object --bucket "$BUCKET_NAME" --key "$key" --version-id "$version" \
                $endpoint_flag $region_flag 2>/dev/null || true
        done
    echo -e "${GREEN}done${NC}"
    
    # Delete all delete markers (if versioning enabled)
    echo -ne "    Deleting delete markers... "
    aws s3api list-object-versions --bucket "$BUCKET_NAME" $endpoint_flag $region_flag \
        --query 'DeleteMarkers[].{Key:Key,VersionId:VersionId}' --output json 2>/dev/null | \
        jq -r '.[] | "\(.Key) \(.VersionId)"' 2>/dev/null | \
        while read key version; do
            [ -n "$key" ] && aws s3api delete-object --bucket "$BUCKET_NAME" --key "$key" --version-id "$version" \
                $endpoint_flag $region_flag 2>/dev/null || true
        done
    echo -e "${GREEN}done${NC}"
    
    # Abort incomplete multipart uploads
    echo -ne "    Aborting multipart uploads... "
    aws s3api list-multipart-uploads --bucket "$BUCKET_NAME" $endpoint_flag $region_flag \
        --query 'Uploads[].{Key:Key,UploadId:UploadId}' --output json 2>/dev/null | \
        jq -r '.[] | "\(.Key) \(.UploadId)"' 2>/dev/null | \
        while read key upload_id; do
            [ -n "$key" ] && aws s3api abort-multipart-upload --bucket "$BUCKET_NAME" --key "$key" --upload-id "$upload_id" \
                $endpoint_flag $region_flag 2>/dev/null || true
        done
    echo -e "${GREEN}done${NC}"
    
    echo -e "  ${GREEN}✓ Bucket cleared completely${NC}"
    
    # Wait for services to detect the empty state
    echo -ne "  Waiting for services to sync (10s)... "
    sleep 10
    echo -e "${GREEN}done${NC}"
}

# Generate random vectors in batches
generate_batch() {
    local start=$1
    local count=$2
    python3 -c "
import json
import random
random.seed($start)
vectors = []
for i in range($start, $start + $count):
    # Generate random unit vector
    v = [random.gauss(0, 1) for _ in range($DIM)]
    norm = sum(x*x for x in v) ** 0.5
    v = [x/norm for x in v]
    vectors.append({
        'id': f'doc-{i}',
        'vector': v,
        'text': f'Document number {i} with random content for testing',
        'attributes': {'batch': $start // $count}
    })
print(json.dumps({'vectors': vectors}))
"
}

# Generate a random vector query
generate_vector_query() {
    python3 -c "
import json
import random
random.seed()
v = [random.gauss(0, 1) for _ in range($DIM)]
norm = sum(x*x for x in v) ** 0.5
v = [x/norm for x in v]
print(json.dumps({'vector': v, 'top_k': 10, 'mode': 'vector'}))
"
}

# Generate a text-only query (BM25)
generate_text_query() {
    python3 -c "
import json
import random
random.seed()
# Random search terms
terms = ['document', 'random', 'content', 'testing', 'number', 'data', 'search', 'query']
query_terms = ' '.join(random.sample(terms, random.randint(1, 3)))
print(json.dumps({'text': query_terms, 'top_k': 10, 'mode': 'text'}))
"
}

# Generate a hybrid query (vector + text)
generate_hybrid_query() {
    python3 -c "
import json
import random
random.seed()
v = [random.gauss(0, 1) for _ in range($DIM)]
norm = sum(x*x for x in v) ** 0.5
v = [x/norm for x in v]
terms = ['document', 'random', 'content', 'testing', 'number', 'data', 'search', 'query']
query_terms = ' '.join(random.sample(terms, random.randint(1, 3)))
print(json.dumps({'vector': v, 'text': query_terms, 'top_k': 10, 'mode': 'hybrid', 'alpha': 0.7}))
"
}

# Measure query latency for a given query type (returns avg ms)
# Args: samples, query_type (vector|text|hybrid)
measure_query_latency() {
    local samples=$1
    local query_type=${2:-vector}
    local total_ms=0
    local success=0
    
    for i in $(seq 1 $samples); do
        case "$query_type" in
            vector) query_payload=$(generate_vector_query) ;;
            text)   query_payload=$(generate_text_query) ;;
            hybrid) query_payload=$(generate_hybrid_query) ;;
            *)      query_payload=$(generate_vector_query) ;;
        esac
        
        start_ms=$(python3 -c "import time; print(int(time.time()*1000))")
        
        response=$(curl -s -w "\n%{http_code}" --max-time 30 \
            -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
            -H "Content-Type: application/json" \
            -d "$query_payload" 2>/dev/null)
        
        end_ms=$(python3 -c "import time; print(int(time.time()*1000))")
        http_code=$(echo "$response" | tail -1)
        
        if [[ "$http_code" == "200" ]]; then
            latency=$((end_ms - start_ms))
            total_ms=$((total_ms + latency))
            success=$((success + 1))
        fi
    done
    
    if [[ $success -gt 0 ]]; then
        avg=$((total_ms / success))
        echo "$avg"
    else
        echo "-1"
    fi
}

# Insert vectors up to target count
insert_vectors() {
    local current=$1
    local target=$2
    local to_insert=$((target - current))
    
    if [[ $to_insert -le 0 ]]; then
        return
    fi
    
    local batches=$(( (to_insert + BATCH_SIZE - 1) / BATCH_SIZE ))
    local inserted=0
    
    echo -ne "  Inserting $to_insert vectors... "
    
    for i in $(seq 0 $((batches - 1))); do
        local batch_start=$((current + i * BATCH_SIZE))
        local batch_count=$BATCH_SIZE
        if [[ $((batch_start + batch_count)) -gt $target ]]; then
            batch_count=$((target - batch_start))
        fi
        
        generate_batch $batch_start $batch_count | \
            curl -s --max-time 60 -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
            -H "Content-Type: application/json" -d @- > /dev/null
        
        inserted=$((inserted + batch_count))
        echo -ne "\r  Inserting $to_insert vectors... $inserted/$to_insert "
    done
    echo -e "${GREEN}done${NC}"
}

# Trigger compaction and wait
trigger_compaction() {
    echo -ne "  Triggering compaction... "
    curl -s --max-time 120 -X POST "$INGEST_URL/v1/namespaces/$NAMESPACE/compact" > /dev/null
    echo -e "${GREEN}done${NC}"
    
    # Wait for compaction to complete and query service to sync
    echo -ne "  Waiting for sync (15s)... "
    sleep 15
    echo -e "${GREEN}done${NC}"
}

# Results storage
declare -a RESULTS_SCALE
declare -a RESULTS_VECTOR_LATENCY
declare -a RESULTS_TEXT_LATENCY
declare -a RESULTS_HYBRID_LATENCY

echo ""

# ═══════════════════════════════════════════════════════════════════
# VECTOR SEARCH TESTS
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}▶ VECTOR SEARCH TESTS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

clear_bucket
current_count=0

for scale in "${SCALES[@]}"; do
    echo -e "  ${YELLOW}[$scale vectors]${NC}"
    
    to_insert=$((scale - current_count))
    if [[ $to_insert -gt 0 ]]; then
        insert_vectors $current_count $scale
        current_count=$scale
        trigger_compaction
    fi
    
    echo -ne "    Measuring latency ($QUERY_SAMPLES queries)... "
    latency=$(measure_query_latency $QUERY_SAMPLES vector)
    if [[ "$latency" == "-1" ]]; then
        echo -e "${RED}failed${NC}"
        latency="N/A"
    else
        echo -e "${GREEN}${latency}ms avg${NC}"
    fi
    
    RESULTS_VECTOR_LATENCY+=("$latency")
    echo ""
done

# ═══════════════════════════════════════════════════════════════════
# TEXT SEARCH TESTS (BM25)
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}▶ TEXT SEARCH TESTS (BM25)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

clear_bucket
current_count=0

for scale in "${SCALES[@]}"; do
    echo -e "  ${YELLOW}[$scale vectors]${NC}"
    
    to_insert=$((scale - current_count))
    if [[ $to_insert -gt 0 ]]; then
        insert_vectors $current_count $scale
        current_count=$scale
        trigger_compaction
    fi
    
    echo -ne "    Measuring latency ($QUERY_SAMPLES queries)... "
    latency=$(measure_query_latency $QUERY_SAMPLES text)
    if [[ "$latency" == "-1" ]]; then
        echo -e "${RED}failed${NC}"
        latency="N/A"
    else
        echo -e "${GREEN}${latency}ms avg${NC}"
    fi
    
    RESULTS_TEXT_LATENCY+=("$latency")
    echo ""
done

# ═══════════════════════════════════════════════════════════════════
# HYBRID SEARCH TESTS (Vector + Text)
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}▶ HYBRID SEARCH TESTS (Vector + Text)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

clear_bucket
current_count=0

for scale in "${SCALES[@]}"; do
    echo -e "  ${YELLOW}[$scale vectors]${NC}"
    
    to_insert=$((scale - current_count))
    if [[ $to_insert -gt 0 ]]; then
        insert_vectors $current_count $scale
        current_count=$scale
        trigger_compaction
    fi
    
    echo -ne "    Measuring latency ($QUERY_SAMPLES queries)... "
    latency=$(measure_query_latency $QUERY_SAMPLES hybrid)
    if [[ "$latency" == "-1" ]]; then
        echo -e "${RED}failed${NC}"
        latency="N/A"
    else
        echo -e "${GREEN}${latency}ms avg${NC}"
    fi
    
    RESULTS_HYBRID_LATENCY+=("$latency")
    echo ""
done

# Populate RESULTS_SCALE (same for all)
RESULTS_SCALE=("${SCALES[@]}")

# Helper to colorize latency
colorize_latency() {
    local val=$1
    if [[ "$val" == "N/A" ]]; then
        echo "${RED}$val${NC}"
    elif [[ $val -lt 100 ]]; then
        echo "${GREEN}$val${NC}"
    elif [[ $val -lt 500 ]]; then
        echo "${YELLOW}$val${NC}"
    else
        echo "${RED}$val${NC}"
    fi
}

# Print summary
echo -e "${BLUE}╔═════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                              Results Summary                                    ║${NC}"
echo -e "${BLUE}╠═════════════════════════════════════════════════════════════════════════════════╣${NC}"
printf "${BLUE}║${NC}  %-10s │ %-15s │ %-15s │ %-15s ${BLUE}║${NC}\n" "Vectors" "Vector (ms)" "Text (ms)" "Hybrid (ms)"
echo -e "${BLUE}╠═════════════════════════════════════════════════════════════════════════════════╣${NC}"

for i in "${!RESULTS_SCALE[@]}"; do
    scale="${RESULTS_SCALE[$i]}"
    vector="${RESULTS_VECTOR_LATENCY[$i]}"
    text="${RESULTS_TEXT_LATENCY[$i]}"
    hybrid="${RESULTS_HYBRID_LATENCY[$i]}"
    
    # Color code each latency
    vector_color=$(colorize_latency "$vector")
    text_color=$(colorize_latency "$text")
    hybrid_color=$(colorize_latency "$hybrid")
    
    printf "${BLUE}║${NC}  %-10s │ " "$scale"
    printf "%-24b │ " "$vector_color"
    printf "%-24b │ " "$text_color"
    printf "%-24b" "$hybrid_color"
    printf " ${BLUE}║${NC}\n"
done

echo -e "${BLUE}╚═════════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Analysis
echo -e "${CYAN}Analysis:${NC}"
echo "  - Vector: HNSW (<10K vectors) or IVF (≥10K vectors) index"
echo "  - Text: BM25 inverted index for full-text search"
echo "  - Hybrid: Combined vector + text with score fusion"
echo ""
echo "  ${GREEN}< 100ms${NC}  = Excellent"
echo "  ${YELLOW}100-500ms${NC} = Acceptable"
echo "  ${RED}> 500ms${NC}   = Consider optimization"
echo ""

# Cleanup note
echo -e "${YELLOW}Note:${NC} Tests used the '$NAMESPACE' namespace."
echo "      Bucket was cleared between test types (vector/text/hybrid)."
echo "      Vectors were inserted incrementally within each test type."
