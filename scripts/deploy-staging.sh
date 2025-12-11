#!/bin/bash
# =============================================================================
# RingRift Staging Deployment Script
# =============================================================================
#
# Deploys the RingRift staging environment with production-like topology.
#
# Usage:
#   ./scripts/deploy-staging.sh [options]
#
# Options:
#   --build         Force rebuild all images
#   --clean         Remove volumes before deploying (fresh start)
#   --skip-health   Skip health check verification
#   --env FILE      Use alternative env file (default: .env.staging)
#   --help          Show this help message
#
# Prerequisites:
#   - Docker and docker-compose installed
#   - .env.staging configured with real secrets (not placeholders)
#   - Ports 3000, 3001, 3002, 5432, 6379, 8001, 9090, 9093 available
#   - At least 8GB RAM available for staging stack
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
FORCE_BUILD=false
CLEAN_START=false
SKIP_HEALTH=false
ENV_FILE=".env.staging"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            FORCE_BUILD=true
            shift
            ;;
        --clean)
            CLEAN_START=true
            shift
            ;;
        --skip-health)
            SKIP_HEALTH=true
            shift
            ;;
        --env)
            ENV_FILE="$2"
            shift 2
            ;;
        --help)
            head -35 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🚀 RingRift Staging Deployment${NC}"
echo "============================================"

# Check script location (should be run from project root)
if [ ! -f "docker-compose.staging.yml" ]; then
    echo -e "${RED}Error: Must be run from project root directory${NC}"
    echo "Expected to find docker-compose.staging.yml in current directory"
    exit 1
fi

# Check prerequisites
echo -e "\n${YELLOW}📋 Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi
echo "  ✓ Docker installed: $(docker --version)"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi
COMPOSE_CMD="docker compose"
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
fi
echo "  ✓ Docker Compose available"

# Check environment file
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: Environment file not found: $ENV_FILE${NC}"
    echo "Please copy .env.staging and configure with real secrets"
    exit 1
fi
echo "  ✓ Environment file: $ENV_FILE"

# Check for placeholder secrets
echo -e "\n${YELLOW}🔐 Validating secrets...${NC}"
if grep -q '<STAGING_' "$ENV_FILE"; then
    echo -e "${RED}Error: Found placeholder secrets in $ENV_FILE${NC}"
    echo "Replace all <STAGING_*> placeholders with real values:"
    grep '<STAGING_' "$ENV_FILE" | head -5
    exit 1
fi

# Check for required secrets
REQUIRED_VARS=("JWT_SECRET" "JWT_REFRESH_SECRET" "DB_PASSWORD")
for var in "${REQUIRED_VARS[@]}"; do
    if ! grep -q "^${var}=" "$ENV_FILE" || grep -q "^${var}=$" "$ENV_FILE"; then
        value=$(grep "^${var}=" "$ENV_FILE" | cut -d'=' -f2)
        if [ -z "$value" ] || [ "$value" = "" ]; then
            echo -e "${RED}Error: Required variable $var is empty in $ENV_FILE${NC}"
            exit 1
        fi
    fi
done
echo "  ✓ Required secrets configured"

# Check available memory
echo -e "\n${YELLOW}💾 Checking resources...${NC}"
if command -v sysctl &> /dev/null; then
    TOTAL_MEM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024/1024/1024}' || echo "8")
elif [ -f /proc/meminfo ]; then
    TOTAL_MEM_GB=$(awk '/MemTotal/ {print $2/1024/1024}' /proc/meminfo)
else
    TOTAL_MEM_GB="8"
fi
echo "  ℹ Available memory: ${TOTAL_MEM_GB}GB (need ~8.5GB for full stack)"

# Check port availability
REQUIRED_PORTS=(3000 3001 3002 5432 6379 8001 9090 9093)
for port in "${REQUIRED_PORTS[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}  ⚠ Port $port is in use${NC}"
    fi
done

# Load environment
echo -e "\n${YELLOW}📦 Loading environment...${NC}"
set -a
source "$ENV_FILE"
set +a
echo "  ✓ Environment loaded"

# Clean start if requested
if [ "$CLEAN_START" = true ]; then
    echo -e "\n${YELLOW}🧹 Cleaning previous deployment...${NC}"
    $COMPOSE_CMD -f docker-compose.staging.yml down -v 2>/dev/null || true
    docker volume prune -f 2>/dev/null || true
    echo "  ✓ Previous volumes removed"
fi

# Build images
echo -e "\n${YELLOW}🔨 Building images...${NC}"
BUILD_ARGS=""
if [ "$FORCE_BUILD" = true ]; then
    BUILD_ARGS="--no-cache"
fi

$COMPOSE_CMD -f docker-compose.staging.yml build $BUILD_ARGS
echo "  ✓ Images built successfully"

# Start infrastructure services first
echo -e "\n${YELLOW}🗄️ Starting infrastructure services...${NC}"
$COMPOSE_CMD -f docker-compose.staging.yml up -d postgres redis

# Wait for database to be ready
echo -e "\n${YELLOW}⏳ Waiting for database...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0
while ! $COMPOSE_CMD -f docker-compose.staging.yml exec -T postgres pg_isready -U "${POSTGRES_USER:-ringrift}" -d "${POSTGRES_DB:-ringrift}" >/dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}Error: Database failed to start after $MAX_RETRIES attempts${NC}"
        echo "Hint: Check logs with: $COMPOSE_CMD -f docker-compose.staging.yml logs postgres --tail 50"
        exit 1
    fi
    echo "  Waiting for PostgreSQL... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done
echo "  ✓ PostgreSQL is ready"

# Wait for Redis
echo -e "\n${YELLOW}⏳ Waiting for Redis...${NC}"
RETRY_COUNT=0
while ! $COMPOSE_CMD -f docker-compose.staging.yml exec -T redis redis-cli ping >/dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}Error: Redis failed to start after $MAX_RETRIES attempts${NC}"
        echo "Hint: Check logs with: $COMPOSE_CMD -f docker-compose.staging.yml logs redis --tail 50"
        exit 1
    fi
    echo "  Waiting for Redis... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done
echo "  ✓ Redis is ready"

# Start AI service
echo -e "\n${YELLOW}🤖 Starting AI service...${NC}"
$COMPOSE_CMD -f docker-compose.staging.yml up -d ai-service

# Wait for AI service (required for staging load tests unless --skip-health is provided)
if [ "$SKIP_HEALTH" = false ]; then
    echo -e "\n${YELLOW}⏳ Waiting for AI service (/health)...${NC}"
    RETRY_COUNT=0
    while ! curl -sf http://localhost:8001/health >/dev/null 2>&1; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo -e "${RED}Error: AI service failed to become healthy after $MAX_RETRIES attempts${NC}"
            echo "Hint: Check AI service logs with: $COMPOSE_CMD -f docker-compose.staging.yml logs ai-service --tail 50"
            exit 1
        fi
        echo "  Waiting for AI service... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    done
    echo "  ✓ AI service is ready"
else
    echo -e "\n${YELLOW}⏭️  Skipping AI service health wait (--skip-health provided).${NC}"
fi

# Start main application
echo -e "\n${YELLOW}🎮 Starting main application...${NC}"
$COMPOSE_CMD -f docker-compose.staging.yml up -d app

# Wait for application (liveness)
echo -e "\n${YELLOW}⏳ Waiting for application...${NC}"
sleep 10
RETRY_COUNT=0
while ! curl -sf http://localhost:3000/health >/dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge 60 ]; then
        echo -e "${RED}Error: Application failed to start after 60 attempts${NC}"
        echo "Checking logs..."
        $COMPOSE_CMD -f docker-compose.staging.yml logs app --tail 50
        exit 1
    fi
    echo "  Waiting for application... ($RETRY_COUNT/60)"
    sleep 2
done
echo "  ✓ Application is responding on /health"

# Readiness check (ensures core dependencies like DB/Redis/AI are wired up)
if [ "$SKIP_HEALTH" = false ]; then
    echo -e "\n${YELLOW}⏳ Verifying application readiness (/ready)...${NC}"
    RETRY_COUNT=0
    while ! curl -sf http://localhost:3000/ready >/dev/null 2>&1; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -ge 60 ]; then
            echo -e "${RED}Error: Application readiness check failed after 60 attempts${NC}"
            echo "Hint: Check service status with:"
            echo "  $COMPOSE_CMD -f docker-compose.staging.yml ps"
            echo "  $COMPOSE_CMD -f docker-compose.staging.yml logs app ai-service postgres redis --tail 50"
            exit 1
        fi
        echo "  Waiting for readiness... ($RETRY_COUNT/60)"
        sleep 2
    done
    echo "  ✓ Application reports ready on /ready"
fi

# Start remaining services
echo -e "\n${YELLOW}📊 Starting monitoring stack...${NC}"
$COMPOSE_CMD -f docker-compose.staging.yml up -d prometheus alertmanager grafana

# Wait a moment for monitoring to initialize
sleep 5

# Start nginx
echo -e "\n${YELLOW}🌐 Starting nginx...${NC}"
$COMPOSE_CMD -f docker-compose.staging.yml up -d nginx 2>/dev/null || echo "  ⚠ Nginx skipped (may require SSL config)"

# Health checks
if [ "$SKIP_HEALTH" = false ]; then
    echo -e "\n${YELLOW}🔍 Running health checks...${NC}"
    
    # Application health
    if curl -sf http://localhost:3000/health >/dev/null 2>&1; then
        echo "  ✓ Application: http://localhost:3000/health"
    else
        echo -e "  ${RED}✗ Application health check failed${NC}"
    fi
    
    # AI service health
    if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
        echo "  ✓ AI Service: http://localhost:8001/health"
    else
        echo -e "  ${YELLOW}⚠ AI Service: not responding (fallback mode)${NC}"
    fi
    
    # Prometheus health
    if curl -sf http://localhost:9090/-/healthy >/dev/null 2>&1; then
        echo "  ✓ Prometheus: http://localhost:9090"
    else
        echo -e "  ${YELLOW}⚠ Prometheus: not responding${NC}"
    fi
    
    # Grafana health
    if curl -sf http://localhost:3002/api/health >/dev/null 2>&1; then
        echo "  ✓ Grafana: http://localhost:3002"
    else
        echo -e "  ${YELLOW}⚠ Grafana: not responding${NC}"
    fi
fi

# Final summary
echo -e "\n${GREEN}✅ Staging environment deployed successfully!${NC}"
echo "============================================"
echo ""
echo "Services:"
echo "  - API:         http://localhost:3000"
echo "  - WebSocket:   ws://localhost:3001"
echo "  - AI Service:  http://localhost:8001"
echo "  - Prometheus:  http://localhost:9090"
echo "  - Grafana:     http://localhost:3002 (admin/${GRAFANA_PASSWORD:-admin})"
echo "  - Alertmanager: http://localhost:9093"
echo ""
echo "Database:"
echo "  - PostgreSQL:  localhost:5432 (user: ringrift)"
echo "  - Redis:       localhost:6379"
echo ""
echo "Load Testing:"
echo "  npm run load:baseline -- -e BASE_URL=http://localhost:3000"
echo "  npm run load:stress -- -e BASE_URL=http://localhost:3000"
echo ""
echo "Teardown:"
echo "  ./scripts/teardown-staging.sh"
echo ""