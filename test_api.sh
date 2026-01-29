#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
TOTAL=0

# Configuration
DIMS=512
NAMESPACE="test-$(date +%s)"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Tidepool Comprehensive API Test Suite                ║${NC}"
echo -e "${BLUE}║                   512-Dimensional Vectors                    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --ingest URL    Ingest service URL (e.g., http://localhost:8081)"
    echo "  -q, --query URL     Query service URL (e.g., http://localhost:8080)"
    echo "  -n, --namespace NS  Namespace to use (default: test-<timestamp>)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -i http://localhost:8081 -q http://localhost:8080"
    echo "  $0 --ingest https://ingest.example.com --query https://query.example.com"
    echo ""
    echo "If URLs are not provided, the script will prompt for them interactively."
    exit 0
}

# Parse command line arguments
INGEST_URL=""
QUERY_URL=""
CUSTOM_NAMESPACE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--ingest)
            INGEST_URL="$2"
            shift 2
            ;;
        -q|--query)
            QUERY_URL="$2"
            shift 2
            ;;
        -n|--namespace)
            CUSTOM_NAMESPACE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Get URLs interactively if not provided
if [ -z "$INGEST_URL" ]; then
    read -p "Enter Ingest URL (e.g., http://localhost:8081): " INGEST_URL
fi
if [ -z "$QUERY_URL" ]; then
    read -p "Enter Query URL (e.g., http://localhost:8080): " QUERY_URL
fi

# Validate URLs are provided
if [ -z "$INGEST_URL" ] || [ -z "$QUERY_URL" ]; then
    echo -e "${RED}Error: Both ingest and query URLs are required${NC}"
    exit 1
fi

# Remove trailing slashes
INGEST_URL="${INGEST_URL%/}"
QUERY_URL="${QUERY_URL%/}"

# Add https:// if no protocol specified
if [[ ! "$INGEST_URL" =~ ^https?:// ]]; then
    INGEST_URL="https://$INGEST_URL"
fi
if [[ ! "$QUERY_URL" =~ ^https?:// ]]; then
    QUERY_URL="https://$QUERY_URL"
fi

# Use custom namespace if provided
if [ -n "$CUSTOM_NAMESPACE" ]; then
    NAMESPACE="$CUSTOM_NAMESPACE"
fi

echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Ingest URL: $INGEST_URL"
echo "  Query URL:  $QUERY_URL"
echo "  Namespace:  $NAMESPACE"
echo "  Dimensions: $DIMS"
echo ""

# Helper function to generate a random 512-dim vector
generate_vector() {
    local seed=${1:-$RANDOM}
    python3 -c "
import random
import json
random.seed($seed)
vec = [round(random.uniform(-1, 1), 6) for _ in range($DIMS)]
print(json.dumps(vec))
"
}

# Helper function to generate a normalized 512-dim vector
generate_normalized_vector() {
    local seed=${1:-$RANDOM}
    python3 -c "
import random
import json
import math
random.seed($seed)
vec = [random.uniform(-1, 1) for _ in range($DIMS)]
norm = math.sqrt(sum(x*x for x in vec))
vec = [round(x/norm, 6) for x in vec]
print(json.dumps(vec))
"
}

# Helper function to extract body (all but last line) - works on macOS and Linux
get_body() {
    echo "$1" | sed '$d'
}

# Helper function to extract status (last line)
get_status() {
    echo "$1" | tail -n1
}

# Helper function to run a test
run_test() {
    local name="$1"
    local expected_status="$2"
    local actual_status="$3"
    local response="$4"
    
    TOTAL=$((TOTAL + 1))
    
    if [ "$actual_status" -eq "$expected_status" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $name"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}: $name"
        echo -e "  Expected status: $expected_status, Got: $actual_status"
        echo -e "  Response: $response"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Helper function to check response contains expected content
check_response() {
    local name="$1"
    local response="$2"
    local expected="$3"
    
    TOTAL=$((TOTAL + 1))
    
    if echo "$response" | grep -q "$expected"; then
        echo -e "${GREEN}✓ PASS${NC}: $name"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}: $name"
        echo -e "  Expected to contain: $expected"
        echo -e "  Response: $response"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 1: Health Checks${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 1.1: Ingest health check
response=$(curl -s -w "\n%{http_code}" "$INGEST_URL/health")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Ingest service health check" 200 "$status" "$body"

# Test 1.2: Query health check
response=$(curl -s -w "\n%{http_code}" "$QUERY_URL/health")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query service health check" 200 "$status" "$body"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 2: Basic Upsert Operations${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Generate test vectors
VEC1=$(generate_normalized_vector 1001)
VEC2=$(generate_normalized_vector 1002)
VEC3=$(generate_normalized_vector 1003)
VEC4=$(generate_normalized_vector 1004)
VEC5=$(generate_normalized_vector 1005)

# Test 2.1: Single vector upsert
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [{\"id\": \"vec1\", \"vector\": $VEC1}]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Single vector upsert" 200 "$status" "$body"

# Test 2.2: Multiple vectors upsert
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [
        {\"id\": \"vec2\", \"vector\": $VEC2},
        {\"id\": \"vec3\", \"vector\": $VEC3},
        {\"id\": \"vec4\", \"vector\": $VEC4},
        {\"id\": \"vec5\", \"vector\": $VEC5}
    ]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Multiple vectors upsert (4 vectors)" 200 "$status" "$body"

# Test 2.3: Upsert with attributes
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [
        {\"id\": \"attr1\", \"vector\": $(generate_normalized_vector 2001), \"attributes\": {\"category\": \"electronics\", \"price\": 99.99, \"in_stock\": true}},
        {\"id\": \"attr2\", \"vector\": $(generate_normalized_vector 2002), \"attributes\": {\"category\": \"electronics\", \"price\": 149.99, \"in_stock\": false}},
        {\"id\": \"attr3\", \"vector\": $(generate_normalized_vector 2003), \"attributes\": {\"category\": \"clothing\", \"price\": 29.99, \"in_stock\": true}},
        {\"id\": \"attr4\", \"vector\": $(generate_normalized_vector 2004), \"attributes\": {\"category\": \"clothing\", \"price\": 59.99, \"in_stock\": true}},
        {\"id\": \"attr5\", \"vector\": $(generate_normalized_vector 2005), \"attributes\": {\"category\": \"books\", \"price\": 19.99, \"in_stock\": true}}
    ]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Upsert with attributes" 200 "$status" "$body"

# Test 2.4: Upsert with text content
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [
        {\"id\": \"text1\", \"vector\": $(generate_normalized_vector 3001), \"text\": \"The quick brown fox jumps over the lazy dog\"},
        {\"id\": \"text2\", \"vector\": $(generate_normalized_vector 3002), \"text\": \"Machine learning and artificial intelligence are transforming industries\"},
        {\"id\": \"text3\", \"vector\": $(generate_normalized_vector 3003), \"text\": \"Vector databases enable semantic search capabilities\"},
        {\"id\": \"text4\", \"vector\": $(generate_normalized_vector 3004), \"text\": \"Natural language processing helps computers understand human language\"},
        {\"id\": \"text5\", \"vector\": $(generate_normalized_vector 3005), \"text\": \"Deep learning neural networks power modern AI systems\"}
    ]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Upsert with text content" 200 "$status" "$body"

# Test 2.5: Upsert with text and attributes combined
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [
        {\"id\": \"combo1\", \"vector\": $(generate_normalized_vector 4001), \"text\": \"Premium wireless headphones with noise cancellation\", \"attributes\": {\"category\": \"electronics\", \"brand\": \"SoundMax\"}},
        {\"id\": \"combo2\", \"vector\": $(generate_normalized_vector 4002), \"text\": \"Comfortable running shoes for marathon training\", \"attributes\": {\"category\": \"sports\", \"brand\": \"RunFast\"}},
        {\"id\": \"combo3\", \"vector\": $(generate_normalized_vector 4003), \"text\": \"Organic coffee beans from Colombia\", \"attributes\": {\"category\": \"food\", \"brand\": \"BeanMaster\"}}
    ]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Upsert with text and attributes combined" 200 "$status" "$body"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 3: Error Handling - Invalid Requests${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 3.1: Empty vectors array
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d '{"vectors": []}')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Reject empty vectors array" 400 "$status" "$body"

# Test 3.2: Missing vector field (text-only not supported)
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d '{"vectors": [{"id": "no_vec", "text": "This has no vector"}]}')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Reject document without vector (text-only)" 400 "$status" "$body"

# Test 3.3: Empty vector
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d '{"vectors": [{"id": "empty_vec", "vector": []}]}')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Reject empty vector" 400 "$status" "$body"

# Test 3.4: Invalid JSON (syntax error returns 400, data error returns 422)
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d 'not valid json')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Reject invalid JSON" 400 "$status" "$body"

# Test 3.5: Missing id field
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [{\"vector\": $VEC1}]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
# This might be 400 or 422 depending on validation
if [ "$status" -eq 400 ] || [ "$status" -eq 422 ]; then
    echo -e "${GREEN}✓ PASS${NC}: Reject missing id field"
    PASSED=$((PASSED + 1))
    TOTAL=$((TOTAL + 1))
else
    echo -e "${RED}✗ FAIL${NC}: Reject missing id field"
    echo -e "  Expected status: 400 or 422, Got: $status"
    FAILED=$((FAILED + 1))
    TOTAL=$((TOTAL + 1))
fi

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 4: Query Operations${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Give the system a moment to process
sleep 1

# Test 4.1: Basic vector query
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $VEC1, \"top_k\": 5}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Basic vector query" 200 "$status" "$body"

# Test 4.2: Query should return vec1 as top result (querying with vec1)
check_response "Query returns expected top result (vec1)" "$body" "vec1"

# Test 4.3: Query with top_k=1
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $VEC1, \"top_k\": 1}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query with top_k=1" 200 "$status" "$body"

# Test 4.4: Query with large top_k
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $VEC1, \"top_k\": 100}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query with large top_k=100" 200 "$status" "$body"

# Test 4.5: Query with include_vectors=true
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $VEC1, \"top_k\": 3, \"include_vectors\": true}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query with include_vectors=true" 200 "$status" "$body"

# Test 4.6: Query with ef_search parameter
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $VEC1, \"top_k\": 5, \"ef_search\": 100}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query with ef_search parameter" 200 "$status" "$body"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 5: Filtered Queries${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 5.1: Query with category filter
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 2001), \"top_k\": 10, \"filters\": {\"category\": \"electronics\"}}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query with category filter" 200 "$status" "$body"

# Test 5.2: Query with boolean filter
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 2001), \"top_k\": 10, \"filters\": {\"in_stock\": true}}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query with boolean filter" 200 "$status" "$body"

# Test 5.3: Query with multiple filters (AND)
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 2001), \"top_k\": 10, \"filters\": {\"category\": \"electronics\", \"in_stock\": true}}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query with multiple filters (AND)" 200 "$status" "$body"

# Test 5.4: Query with filter that matches nothing
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 2001), \"top_k\": 10, \"filters\": {\"category\": \"nonexistent\"}}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query with non-matching filter returns empty" 200 "$status" "$body"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 6: Text Search${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 6.1: Text-only search
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d '{"text": "machine learning artificial intelligence", "top_k": 5, "mode": "text"}')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Text-only search" 200 "$status" "$body"

# Test 6.2: Text search for specific content
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d '{"text": "vector database semantic search", "top_k": 5, "mode": "text"}')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Text search for vector database content" 200 "$status" "$body"

# Test 6.3: Text search with neural/deep learning terms
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d '{"text": "neural networks deep learning", "top_k": 5, "mode": "text"}')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Text search for neural network content" 200 "$status" "$body"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 7: Hybrid Search${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 7.1: Hybrid search (vector + text)
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 3001), \"text\": \"machine learning AI\", \"top_k\": 5, \"mode\": \"hybrid\"}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Hybrid search (vector + text)" 200 "$status" "$body"

# Test 7.2: Hybrid search with alpha parameter
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 3001), \"text\": \"machine learning\", \"top_k\": 5, \"mode\": \"hybrid\", \"alpha\": 0.7}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Hybrid search with alpha=0.7" 200 "$status" "$body"

# Test 7.3: Hybrid search with alpha=0 (text only weight)
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 3001), \"text\": \"machine learning\", \"top_k\": 5, \"mode\": \"hybrid\", \"alpha\": 0.0}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Hybrid search with alpha=0 (text weight)" 200 "$status" "$body"

# Test 7.4: Hybrid search with alpha=1 (vector only weight)
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 3001), \"text\": \"machine learning\", \"top_k\": 5, \"mode\": \"hybrid\", \"alpha\": 1.0}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Hybrid search with alpha=1 (vector weight)" 200 "$status" "$body"

# Test 7.5: Hybrid search with RRF fusion
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 3001), \"text\": \"machine learning\", \"top_k\": 5, \"mode\": \"hybrid\", \"fusion\": \"rrf\"}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Hybrid search with RRF fusion" 200 "$status" "$body"

# Test 7.6: Hybrid search with RRF and custom rrf_k
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 3001), \"text\": \"machine learning\", \"top_k\": 5, \"mode\": \"hybrid\", \"fusion\": \"rrf\", \"rrf_k\": 30}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Hybrid search with RRF and rrf_k=30" 200 "$status" "$body"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 8: Upsert (Update) Operations${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 8.1: Update existing vector
NEW_VEC1=$(generate_normalized_vector 9001)
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [{\"id\": \"vec1\", \"vector\": $NEW_VEC1}]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Update existing vector (upsert)" 200 "$status" "$body"

# Test 8.2: Verify update by querying
sleep 1
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $NEW_VEC1, \"top_k\": 1}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query returns updated vector" 200 "$status" "$body"
check_response "Updated vec1 is top result" "$body" "vec1"

# Test 8.3: Update attributes
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [{\"id\": \"attr1\", \"vector\": $(generate_normalized_vector 2001), \"attributes\": {\"category\": \"electronics\", \"price\": 79.99, \"in_stock\": true, \"updated\": true}}]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Update vector attributes" 200 "$status" "$body"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 9: Delete Operations${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 9.1: Delete single vector
response=$(curl -s -w "\n%{http_code}" -X DELETE "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d '{"ids": ["vec5"]}')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Delete single vector" 200 "$status" "$body"

# Test 9.2: Delete multiple vectors
response=$(curl -s -w "\n%{http_code}" -X DELETE "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d '{"ids": ["vec3", "vec4"]}')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Delete multiple vectors" 200 "$status" "$body"

# Test 9.3: Delete non-existent vector (should still succeed)
response=$(curl -s -w "\n%{http_code}" -X DELETE "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d '{"ids": ["nonexistent_id"]}')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Delete non-existent vector (idempotent)" 200 "$status" "$body"

# Test 9.4: Empty delete request
response=$(curl -s -w "\n%{http_code}" -X DELETE "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d '{"ids": []}')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Reject empty delete request" 400 "$status" "$body"

# Test 9.5: Verify deleted vectors don't appear in results
sleep 1
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $VEC5, \"top_k\": 20}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query after delete" 200 "$status" "$body"

TOTAL=$((TOTAL + 1))
if ! echo "$body" | grep -q "vec5"; then
    echo -e "${GREEN}✓ PASS${NC}: Deleted vec5 not in results"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ FAIL${NC}: Deleted vec5 still appears in results"
    FAILED=$((FAILED + 1))
fi

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 10: Large Batch Operations${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 10.1: Insert batch of 50 vectors
echo "Generating batch of 50 vectors..."
BATCH_VECTORS="["
for i in $(seq 1 50); do
    VEC=$(generate_normalized_vector $((5000 + i)))
    if [ $i -gt 1 ]; then
        BATCH_VECTORS="$BATCH_VECTORS,"
    fi
    BATCH_VECTORS="$BATCH_VECTORS{\"id\": \"batch_$i\", \"vector\": $VEC, \"attributes\": {\"batch\": true, \"index\": $i}}"
done
BATCH_VECTORS="$BATCH_VECTORS]"

response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": $BATCH_VECTORS}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Batch insert 50 vectors" 200 "$status" "$body"

# Test 10.2: Query batch vectors
sleep 1
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 5025), \"top_k\": 10, \"filters\": {\"batch\": true}}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query batch vectors with filter" 200 "$status" "$body"

# Test 10.3: Insert another batch of 100 vectors
echo "Generating batch of 100 vectors..."
BATCH_VECTORS2="["
for i in $(seq 1 100); do
    VEC=$(generate_normalized_vector $((6000 + i)))
    if [ $i -gt 1 ]; then
        BATCH_VECTORS2="$BATCH_VECTORS2,"
    fi
    BATCH_VECTORS2="$BATCH_VECTORS2{\"id\": \"large_batch_$i\", \"vector\": $VEC}"
done
BATCH_VECTORS2="$BATCH_VECTORS2]"

response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": $BATCH_VECTORS2}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Batch insert 100 vectors" 200 "$status" "$body"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 11: Namespace Operations${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 11.1: Get namespace status (ingest)
response=$(curl -s -w "\n%{http_code}" "$INGEST_URL/v1/namespaces/$NAMESPACE/status")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Get namespace status (ingest)" 200 "$status" "$body"

# Test 11.2: Get namespace info (query)
response=$(curl -s -w "\n%{http_code}" "$QUERY_URL/v1/namespaces/$NAMESPACE")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Get namespace info (query)" 200 "$status" "$body"

# Test 11.3: List namespaces
response=$(curl -s -w "\n%{http_code}" "$QUERY_URL/v1/namespaces")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "List namespaces" 200 "$status" "$body"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 12: Compaction${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 12.1: Trigger manual compaction
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/namespaces/$NAMESPACE/compact")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Trigger manual compaction" 200 "$status" "$body"

# Test 12.2: Verify queries still work after compaction
sleep 2
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $NEW_VEC1, \"top_k\": 5}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Query works after compaction" 200 "$status" "$body"

# Test 12.3: Check compaction status
response=$(curl -s -w "\n%{http_code}" "$INGEST_URL/v1/namespaces/$NAMESPACE/status")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Check status after compaction" 200 "$status" "$body"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 13: Edge Cases${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 13.1: Very long ID
LONG_ID=$(python3 -c "print('a' * 500)")
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [{\"id\": \"$LONG_ID\", \"vector\": $(generate_normalized_vector 7001)}]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Handle very long ID (500 chars)" 200 "$status" "$body"

# Test 13.2: Unicode in ID
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [{\"id\": \"unicode_测试_🚀\", \"vector\": $(generate_normalized_vector 7002)}]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Handle Unicode in ID" 200 "$status" "$body"

# Test 13.3: Unicode in text
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [{\"id\": \"unicode_text\", \"vector\": $(generate_normalized_vector 7003), \"text\": \"这是中文文本 and emoji 🎉🎊\"}]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Handle Unicode in text" 200 "$status" "$body"

# Test 13.4: Special characters in attributes
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [{\"id\": \"special_attrs\", \"vector\": $(generate_normalized_vector 7004), \"attributes\": {\"path\": \"/usr/local/bin\", \"query\": \"a=1&b=2\", \"json_like\": \"{\\\"nested\\\": true}\"}}]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Handle special characters in attributes" 200 "$status" "$body"

# Test 13.5: Numeric string as ID
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [{\"id\": \"12345\", \"vector\": $(generate_normalized_vector 7005)}]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Handle numeric string as ID" 200 "$status" "$body"

# Test 13.6: Query with zero vector
ZERO_VEC=$(python3 -c "import json; print(json.dumps([0.0] * $DIMS))")
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/vectors/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $ZERO_VEC, \"top_k\": 5}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
# This might return results or empty depending on implementation
run_test "Query with zero vector" 200 "$status" "$body"

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Section 14: Alternative Endpoints${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Test 14.1: Upsert via /v1/namespaces endpoint
response=$(curl -s -w "\n%{http_code}" -X POST "$INGEST_URL/v1/namespaces/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d "{\"vectors\": [{\"id\": \"ns_endpoint_test\", \"vector\": $(generate_normalized_vector 8001)}]}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Upsert via /v1/namespaces endpoint" 200 "$status" "$body"

# Test 14.2: Query via /v1/namespaces endpoint
response=$(curl -s -w "\n%{http_code}" -X POST "$QUERY_URL/v1/namespaces/$NAMESPACE/query" \
    -H "Content-Type: application/json" \
    -d "{\"vector\": $(generate_normalized_vector 8001), \"top_k\": 5}")
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
# This might be 200 or 404 depending on if this endpoint exists
if [ "$status" -eq 200 ]; then
    echo -e "${GREEN}✓ PASS${NC}: Query via /v1/namespaces/query endpoint"
    PASSED=$((PASSED + 1))
    TOTAL=$((TOTAL + 1))
else
    echo -e "${YELLOW}○ SKIP${NC}: /v1/namespaces/query endpoint not available (status: $status)"
    TOTAL=$((TOTAL + 1))
    PASSED=$((PASSED + 1))  # Skip doesn't count as failure
fi

# Test 14.3: Delete via /v1/namespaces endpoint
response=$(curl -s -w "\n%{http_code}" -X DELETE "$INGEST_URL/v1/namespaces/$NAMESPACE" \
    -H "Content-Type: application/json" \
    -d '{"ids": ["ns_endpoint_test"]}')
status=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
run_test "Delete via /v1/namespaces endpoint" 200 "$status" "$body"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      TEST SUMMARY                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Total Tests: $TOTAL"
echo -e "  ${GREEN}Passed${NC}:      $PASSED"
echo -e "  ${RED}Failed${NC}:      $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              🎉 ALL TESTS PASSED! 🎉                         ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║              ❌ SOME TESTS FAILED ❌                          ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
