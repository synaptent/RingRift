#!/bin/bash
#
# Install Cloud Provider CLI Tools (December 2025)
#
# Installs CLI tools for all cloud providers used in the RingRift cluster.
# These tools enable NodeAvailabilityDaemon to query instance states directly.
#
# Providers:
# - Lambda Labs: lambda-cloud-client
# - Vast.ai: vastai
# - RunPod: runpodctl
# - Vultr: vultr-cli (optional)
# - Hetzner: hcloud (optional, for CPU nodes)
# - Nebius: yc (Yandex Cloud CLI)
#
# Usage:
#   ./scripts/install_provider_clis.sh
#   ./scripts/install_provider_clis.sh --check  # Check what's installed
#
# Required Environment Variables:
#   LAMBDA_API_KEY - Lambda Labs API key
#   VAST_API_KEY - Vast.ai API key
#   RUNPOD_API_KEY - RunPod API key
#   VULTR_API_KEY - Vultr API key (optional)
#   HCLOUD_TOKEN - Hetzner Cloud token (optional)
#   NEBIUS_TOKEN - Nebius/Yandex Cloud token (optional)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running in check mode
CHECK_ONLY=false
if [[ "$1" == "--check" ]]; then
    CHECK_ONLY=true
fi

# ============================================================================
# Check Functions
# ============================================================================

check_python() {
    if command -v python3 &> /dev/null; then
        info "Python3: $(python3 --version)"
        return 0
    else
        error "Python3 not found"
        return 1
    fi
}

check_pip() {
    if command -v pip3 &> /dev/null || python3 -m pip --version &> /dev/null; then
        info "pip: available"
        return 0
    else
        warn "pip not found"
        return 1
    fi
}

check_lambda_cli() {
    if python3 -c "import lambda_cloud" 2>/dev/null; then
        info "Lambda Cloud CLI: installed"
        return 0
    else
        warn "Lambda Cloud CLI: not installed"
        return 1
    fi
}

check_vast_cli() {
    if command -v vastai &> /dev/null; then
        info "Vast.ai CLI: $(vastai --version 2>/dev/null || echo 'installed')"
        return 0
    else
        warn "Vast.ai CLI: not installed"
        return 1
    fi
}

check_runpod_cli() {
    if command -v runpodctl &> /dev/null; then
        info "RunPod CLI: installed"
        return 0
    elif python3 -c "import runpod" 2>/dev/null; then
        info "RunPod Python SDK: installed"
        return 0
    else
        warn "RunPod CLI: not installed"
        return 1
    fi
}

check_vultr_cli() {
    if command -v vultr-cli &> /dev/null; then
        info "Vultr CLI: installed"
        return 0
    elif python3 -c "import vultr" 2>/dev/null; then
        info "Vultr Python SDK: installed"
        return 0
    else
        warn "Vultr CLI: not installed (optional)"
        return 1
    fi
}

check_hcloud() {
    if command -v hcloud &> /dev/null; then
        info "Hetzner hcloud: $(hcloud version 2>/dev/null || echo 'installed')"
        return 0
    else
        warn "Hetzner hcloud: not installed (optional)"
        return 1
    fi
}

check_nebius() {
    if command -v yc &> /dev/null; then
        info "Yandex Cloud CLI (yc): installed"
        return 0
    else
        warn "Yandex Cloud CLI: not installed (optional)"
        return 1
    fi
}

check_all() {
    echo "========================================"
    echo "Cloud Provider CLI Status Check"
    echo "========================================"
    echo ""

    check_python
    check_pip
    echo ""

    echo "--- Provider CLIs ---"
    check_lambda_cli || true
    check_vast_cli || true
    check_runpod_cli || true
    check_vultr_cli || true
    check_hcloud || true
    check_nebius || true
    echo ""

    echo "--- API Key Status ---"
    if [[ -n "$LAMBDA_API_KEY" ]]; then
        info "LAMBDA_API_KEY: set"
    else
        warn "LAMBDA_API_KEY: not set"
    fi

    if [[ -n "$VAST_API_KEY" ]] || [[ -f ~/.config/vastai/vast_api_key ]] || [[ -f ~/.vastai_api_key ]]; then
        info "VAST_API_KEY: set"
    else
        warn "VAST_API_KEY: not set"
    fi

    if [[ -n "$RUNPOD_API_KEY" ]]; then
        info "RUNPOD_API_KEY: set"
    else
        warn "RUNPOD_API_KEY: not set"
    fi

    if [[ -n "$VULTR_API_KEY" ]]; then
        info "VULTR_API_KEY: set"
    else
        warn "VULTR_API_KEY: not set (optional)"
    fi

    if [[ -n "$HCLOUD_TOKEN" ]]; then
        info "HCLOUD_TOKEN: set"
    else
        warn "HCLOUD_TOKEN: not set (optional)"
    fi

    if [[ -n "$NEBIUS_TOKEN" ]]; then
        info "NEBIUS_TOKEN: set"
    else
        warn "NEBIUS_TOKEN: not set (optional)"
    fi

    echo ""
    echo "========================================"
}

# ============================================================================
# Install Functions
# ============================================================================

install_lambda_cli() {
    info "Installing Lambda Cloud CLI..."
    pip3 install --user lambda-cloud-client
    info "Lambda Cloud CLI installed"
}

install_vast_cli() {
    info "Installing Vast.ai CLI..."
    pip3 install --user vastai
    info "Vast.ai CLI installed"

    # Set up API key if available
    if [[ -n "$VAST_API_KEY" ]]; then
        mkdir -p ~/.config/vastai
        echo "$VAST_API_KEY" > ~/.config/vastai/vast_api_key
        info "Vast.ai API key configured"
    fi
}

install_runpod_cli() {
    info "Installing RunPod CLI..."
    pip3 install --user runpod

    # Also install runpodctl if on Linux
    if [[ "$(uname)" == "Linux" ]]; then
        info "Installing runpodctl binary..."
        curl -sSL https://raw.githubusercontent.com/runpod/runpodctl/main/install.sh | bash
    fi
    info "RunPod CLI installed"
}

install_vultr_cli() {
    info "Installing Vultr Python SDK..."
    pip3 install --user vultr-python
    info "Vultr Python SDK installed"
}

install_hcloud() {
    info "Installing Hetzner hcloud CLI..."

    if [[ "$(uname)" == "Darwin" ]]; then
        brew install hcloud
    elif [[ "$(uname)" == "Linux" ]]; then
        # Download latest release
        HCLOUD_VERSION=$(curl -s https://api.github.com/repos/hetznercloud/cli/releases/latest | grep tag_name | cut -d '"' -f 4)
        wget -q "https://github.com/hetznercloud/cli/releases/download/${HCLOUD_VERSION}/hcloud-linux-amd64.tar.gz" -O /tmp/hcloud.tar.gz
        tar -xzf /tmp/hcloud.tar.gz -C /tmp
        mv /tmp/hcloud /usr/local/bin/ 2>/dev/null || sudo mv /tmp/hcloud /usr/local/bin/
        rm /tmp/hcloud.tar.gz
    fi
    info "Hetzner hcloud installed"
}

install_nebius() {
    info "Installing Yandex Cloud CLI..."
    curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash
    info "Yandex Cloud CLI installed"
    warn "Run 'yc init' to configure"
}

install_all() {
    echo "========================================"
    echo "Installing Cloud Provider CLI Tools"
    echo "========================================"
    echo ""

    # Check prerequisites
    if ! check_python; then
        error "Python3 is required. Please install Python3 first."
        exit 1
    fi

    if ! check_pip; then
        error "pip is required. Please install pip first."
        exit 1
    fi

    echo ""
    echo "--- Installing Provider CLIs ---"
    echo ""

    # Core providers (required for full cluster visibility)
    install_lambda_cli
    install_vast_cli
    install_runpod_cli

    # Optional providers
    if [[ -n "$VULTR_API_KEY" ]]; then
        install_vultr_cli
    else
        info "Skipping Vultr CLI (no VULTR_API_KEY)"
    fi

    if [[ -n "$HCLOUD_TOKEN" ]]; then
        install_hcloud
    else
        info "Skipping Hetzner hcloud (no HCLOUD_TOKEN)"
    fi

    if [[ -n "$NEBIUS_TOKEN" ]]; then
        install_nebius
    else
        info "Skipping Nebius CLI (no NEBIUS_TOKEN)"
    fi

    echo ""
    echo "========================================"
    echo "Installation Complete!"
    echo "========================================"
    echo ""
    echo "Required environment variables:"
    echo "  export LAMBDA_API_KEY='your-lambda-api-key'"
    echo "  export VAST_API_KEY='your-vast-api-key'"
    echo "  export RUNPOD_API_KEY='your-runpod-api-key'"
    echo ""
    echo "Optional (for full provider visibility):"
    echo "  export VULTR_API_KEY='your-vultr-api-key'"
    echo "  export HCLOUD_TOKEN='your-hetzner-token'"
    echo "  export NEBIUS_TOKEN='your-nebius-token'"
    echo ""
    echo "Run with --check to verify installation:"
    echo "  ./scripts/install_provider_clis.sh --check"
}

# ============================================================================
# Main
# ============================================================================

if [[ "$CHECK_ONLY" == "true" ]]; then
    check_all
else
    install_all
fi
