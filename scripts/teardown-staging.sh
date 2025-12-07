#!/bin/bash
# =============================================================================
# RingRift Staging Teardown Script
# =============================================================================
#
# Stops and removes the RingRift staging environment.
#
# Usage:
#   ./scripts/teardown-staging.sh [options]
#
# Options:
#   --volumes, -v   Also remove data volumes (WARNING: destroys all data)
#   --images, -i    Also remove Docker images
#   --all           Remove everything including volumes and images
#   --force, -f     Skip confirmation prompts
#   --help          Show this help message
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
REMOVE_VOLUMES=false
REMOVE_IMAGES=false
FORCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --volumes|-v)
            REMOVE_VOLUMES=true
            shift
            ;;
        --images|-i)
            REMOVE_IMAGES=true
            shift
            ;;
        --all)
            REMOVE_VOLUMES=true
            REMOVE_IMAGES=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --help)
            head -20 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🛑 RingRift Staging Teardown${NC}"
echo "============================================"

# Check script location (should be run from project root)
if [ ! -f "docker-compose.staging.yml" ]; then
    echo -e "${RED}Error: Must be run from project root directory${NC}"
    echo "Expected to find docker-compose.staging.yml in current directory"
    exit 1
fi

# Determine compose command
COMPOSE_CMD="docker compose"
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
fi

# Show current status
echo -e "\n${YELLOW}📊 Current staging services:${NC}"
$COMPOSE_CMD -f docker-compose.staging.yml ps 2>/dev/null || echo "  No services running"

# Confirmation for destructive operations
if [ "$REMOVE_VOLUMES" = true ] && [ "$FORCE" = false ]; then
    echo ""
    echo -e "${RED}⚠️  WARNING: This will destroy all staging data including:${NC}"
    echo "  - PostgreSQL database data"
    echo "  - Redis cache data"
    echo "  - Prometheus metrics data"
    echo "  - Grafana dashboards and settings"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi
fi

# Stop services
echo -e "\n${YELLOW}⏹️  Stopping services...${NC}"
$COMPOSE_CMD -f docker-compose.staging.yml stop 2>/dev/null || true
echo "  ✓ Services stopped"

# Remove containers
echo -e "\n${YELLOW}🗑️  Removing containers...${NC}"
if [ "$REMOVE_VOLUMES" = true ]; then
    $COMPOSE_CMD -f docker-compose.staging.yml down -v 2>/dev/null || true
    echo "  ✓ Containers and volumes removed"
else
    $COMPOSE_CMD -f docker-compose.staging.yml down 2>/dev/null || true
    echo "  ✓ Containers removed (volumes preserved)"
fi

# Remove images if requested
if [ "$REMOVE_IMAGES" = true ]; then
    echo -e "\n${YELLOW}🖼️  Removing images...${NC}"
    
    # Get project name prefix
    PROJECT_NAME=$(basename "$(pwd)")
    
    # Remove project-specific images
    docker images --filter "label=com.ringrift.environment=staging" -q 2>/dev/null | xargs -r docker rmi -f 2>/dev/null || true
    
    # Also try to remove by name pattern
    docker images | grep -E "^${PROJECT_NAME}.*staging" | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true
    docker images | grep -E "ringrift.*staging" | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true
    
    echo "  ✓ Staging images removed"
fi

# Clean up networks
echo -e "\n${YELLOW}🔌 Cleaning up networks...${NC}"
docker network rm staging-network 2>/dev/null || true
echo "  ✓ Networks cleaned up"

# Prune orphaned resources
echo -e "\n${YELLOW}🧹 Pruning orphaned resources...${NC}"
docker container prune -f --filter "label=com.ringrift.environment=staging" 2>/dev/null || true
echo "  ✓ Orphaned containers pruned"

# Final status
echo -e "\n${GREEN}✅ Staging environment teardown complete!${NC}"
echo "============================================"

if [ "$REMOVE_VOLUMES" = true ]; then
    echo "  All staging data has been removed."
    echo "  Run ./scripts/deploy-staging.sh --clean to start fresh."
else
    echo "  Data volumes have been preserved."
    echo "  Run ./scripts/deploy-staging.sh to restart."
fi

echo ""