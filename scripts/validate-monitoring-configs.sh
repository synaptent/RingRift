#!/bin/bash
# =============================================================================
# RingRift Monitoring Configuration Validator
# =============================================================================
#
# This script validates Prometheus and Alertmanager configurations to catch
# misconfigurations before deployment. It uses official Prometheus tools via
# Docker containers for accurate validation.
#
# Usage:
#   ./scripts/validate-monitoring-configs.sh        # Run all validations
#   ./scripts/validate-monitoring-configs.sh --help # Show help
#
# Requirements:
#   - Docker (preferred) for full validation with official tools
#   - OR: Python 3 with pyyaml for basic YAML validation fallback
#
# Exit codes:
#   0 - All validations passed
#   1 - Validation failed
#   2 - Required tools not available
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration paths (relative to project root)
PROMETHEUS_CONFIG="monitoring/prometheus/prometheus.yml"
PROMETHEUS_ALERTS="monitoring/prometheus/alerts.yml"
ALERTMANAGER_CONFIG="monitoring/alertmanager/alertmanager.yml"

# Docker images
PROMETHEUS_IMAGE="prom/prometheus:latest"
ALERTMANAGER_IMAGE="prom/alertmanager:latest"

# Track validation results
VALIDATION_ERRORS=0

# =============================================================================
# Utility Functions
# =============================================================================

print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

show_help() {
    cat << EOF
RingRift Monitoring Configuration Validator

Usage:
    ./scripts/validate-monitoring-configs.sh [OPTIONS]

Options:
    --help, -h      Show this help message
    --docker-only   Only use Docker-based validation (fail if Docker unavailable)
    --yaml-only     Only perform basic YAML validation (no Docker required)
    --verbose, -v   Show verbose output

Description:
    This script validates Prometheus and Alertmanager configuration files
    to catch syntax errors, invalid PromQL queries, and misconfigurations
    before deployment.

Validation Checks:
    1. YAML syntax validation
    2. Prometheus config syntax (promtool check config)
    3. Prometheus alert rules (promtool check rules)
    4. Alertmanager config (amtool check-config)
    5. Common configuration issues

Files Validated:
    - ${PROMETHEUS_CONFIG}
    - ${PROMETHEUS_ALERTS}
    - ${ALERTMANAGER_CONFIG}

Examples:
    ./scripts/validate-monitoring-configs.sh
    ./scripts/validate-monitoring-configs.sh --docker-only
    ./scripts/validate-monitoring-configs.sh --yaml-only --verbose

EOF
    exit 0
}

# =============================================================================
# Validation Methods
# =============================================================================

check_file_exists() {
    local file="$1"
    local description="$2"
    
    if [[ -f "$file" ]]; then
        print_success "Found $description: $file"
        return 0
    else
        print_error "Missing $description: $file"
        return 1
    fi
}

# Basic YAML validation using Python
validate_yaml_syntax() {
    local file="$1"
    local description="$2"
    
    print_info "Validating YAML syntax: $description"
    
    # Try Python first
    if command -v python3 &> /dev/null; then
        if python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
            print_success "YAML syntax valid: $file"
            return 0
        else
            print_error "YAML syntax error in $file"
            python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>&1 || true
            return 1
        fi
    elif command -v python &> /dev/null; then
        if python -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
            print_success "YAML syntax valid: $file"
            return 0
        else
            print_error "YAML syntax error in $file"
            return 1
        fi
    else
        print_warning "Python not available for YAML validation, skipping basic syntax check"
        return 0
    fi
}

# Docker-based Prometheus config validation
validate_prometheus_config_docker() {
    local config_file="$1"
    
    print_info "Validating Prometheus config with promtool (Docker)..."
    
    # Create a temporary directory for the config
    local temp_dir
    temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    # Copy config files to temp directory
    cp "$config_file" "$temp_dir/prometheus.yml"
    
    # Copy alerts file if referenced and exists
    if [[ -f "$PROMETHEUS_ALERTS" ]]; then
        mkdir -p "$temp_dir/alerts"
        cp "$PROMETHEUS_ALERTS" "$temp_dir/alerts.yml"
        # Update the rule_files path in the copied config
        sed -i.bak "s|/etc/prometheus/alerts.yml|/config/alerts.yml|g" "$temp_dir/prometheus.yml" 2>/dev/null || \
        sed -i '' "s|/etc/prometheus/alerts.yml|/config/alerts.yml|g" "$temp_dir/prometheus.yml"
    fi
    
    # Run promtool check config
    local output
    if output=$(docker run --rm -v "$temp_dir:/config:ro" "$PROMETHEUS_IMAGE" \
        promtool check config /config/prometheus.yml 2>&1); then
        print_success "Prometheus config is valid"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "$output"
        fi
        return 0
    else
        print_error "Prometheus config validation failed"
        echo "$output"
        return 1
    fi
}

# Docker-based Prometheus rules validation
validate_prometheus_rules_docker() {
    local rules_file="$1"
    
    print_info "Validating Prometheus alert rules with promtool (Docker)..."
    
    # Create temp directory
    local temp_dir
    temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    cp "$rules_file" "$temp_dir/alerts.yml"
    
    # Run promtool check rules
    local output
    if output=$(docker run --rm -v "$temp_dir:/config:ro" "$PROMETHEUS_IMAGE" \
        promtool check rules /config/alerts.yml 2>&1); then
        print_success "Prometheus alert rules are valid"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "$output"
        fi
        return 0
    else
        print_error "Prometheus alert rules validation failed"
        echo "$output"
        return 1
    fi
}

# Docker-based Alertmanager config validation
validate_alertmanager_config_docker() {
    local config_file="$1"
    
    print_info "Validating Alertmanager config with amtool (Docker)..."
    
    # Create temp directory
    local temp_dir
    temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    cp "$config_file" "$temp_dir/alertmanager.yml"
    
    # Run amtool check-config
    local output
    if output=$(docker run --rm -v "$temp_dir:/config:ro" "$ALERTMANAGER_IMAGE" \
        amtool check-config /config/alertmanager.yml 2>&1); then
        print_success "Alertmanager config is valid"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "$output"
        fi
        return 0
    else
        print_error "Alertmanager config validation failed"
        echo "$output"
        return 1
    fi
}

# Check for common configuration issues
check_common_issues() {
    local file="$1"
    local description="$2"
    
    print_info "Checking for common issues in $description..."
    
    local issues_found=0
    
    # Check for tabs (YAML prefers spaces)
    if grep -P '\t' "$file" > /dev/null 2>&1; then
        print_warning "Found tabs in $file (YAML prefers spaces for indentation)"
    fi
    
    # Check for trailing whitespace
    if grep -E '\s+$' "$file" > /dev/null 2>&1; then
        print_warning "Found trailing whitespace in $file"
    fi
    
    # Check for hardcoded secrets (basic patterns)
    if grep -iE "(password|secret|token|key):\s*['\"]?[a-zA-Z0-9]{16,}['\"]?" "$file" 2>/dev/null | grep -v '\${' > /dev/null; then
        print_warning "Potential hardcoded secrets found in $file (review manually)"
        issues_found=1
    fi
    
    # Prometheus-specific checks
    if [[ "$file" == *"prometheus"* ]]; then
        # Check for scrape_interval less than 10s (can cause high load)
        if grep -E "scrape_interval:\s*[0-9]s" "$file" | grep -v "scrape_interval:\s*[1-9][0-9]" > /dev/null 2>&1; then
            print_warning "Very short scrape_interval detected in $file (< 10s can cause high load)"
        fi
    fi
    
    # Alertmanager-specific checks
    if [[ "$file" == *"alertmanager"* ]]; then
        # Check for very short group_wait (can cause alert storms)
        if grep -E "group_wait:\s*[0-5]s" "$file" > /dev/null 2>&1; then
            print_warning "Very short group_wait in $file (< 5s may cause alert storms)"
        fi
    fi
    
    if [[ $issues_found -eq 0 ]]; then
        print_success "No common issues found in $description"
    fi
    
    return 0
}

# =============================================================================
# Main Validation Runner
# =============================================================================

run_yaml_validation() {
    print_header "YAML Syntax Validation"
    
    validate_yaml_syntax "$PROMETHEUS_CONFIG" "Prometheus config" || true
    validate_yaml_syntax "$PROMETHEUS_ALERTS" "Prometheus alerts" || true
    validate_yaml_syntax "$ALERTMANAGER_CONFIG" "Alertmanager config" || true
}

run_docker_validation() {
    print_header "Docker-based Validation (Official Tools)"
    
    if ! command -v docker &> /dev/null; then
        if [[ "$DOCKER_ONLY" == "true" ]]; then
            print_error "Docker is required for --docker-only mode but is not available"
            exit 2
        else
            print_warning "Docker not available, skipping official tool validation"
            print_info "Install Docker for full validation with promtool and amtool"
            return 0
        fi
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        if [[ "$DOCKER_ONLY" == "true" ]]; then
            print_error "Docker daemon is not running"
            exit 2
        else
            print_warning "Docker daemon not running, skipping official tool validation"
            return 0
        fi
    fi
    
    print_info "Pulling Prometheus image (if needed)..."
    docker pull "$PROMETHEUS_IMAGE" > /dev/null 2>&1 || true
    
    print_info "Pulling Alertmanager image (if needed)..."
    docker pull "$ALERTMANAGER_IMAGE" > /dev/null 2>&1 || true
    
    # Run validations
    validate_prometheus_config_docker "$PROMETHEUS_CONFIG" || true
    validate_prometheus_rules_docker "$PROMETHEUS_ALERTS" || true
    validate_alertmanager_config_docker "$ALERTMANAGER_CONFIG" || true
}

run_common_checks() {
    print_header "Common Configuration Checks"
    
    check_common_issues "$PROMETHEUS_CONFIG" "Prometheus config"
    check_common_issues "$PROMETHEUS_ALERTS" "Prometheus alerts"
    check_common_issues "$ALERTMANAGER_CONFIG" "Alertmanager config"
}

# =============================================================================
# Entry Point
# =============================================================================

main() {
    # Parse arguments
    VERBOSE=false
    DOCKER_ONLY=false
    YAML_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --docker-only)
                DOCKER_ONLY=true
                shift
                ;;
            --yaml-only)
                YAML_ONLY=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                ;;
        esac
    done
    
    # Header
    echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       RingRift Monitoring Configuration Validator               ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
    
    # Check that config files exist
    print_header "Checking Configuration Files"
    
    check_file_exists "$PROMETHEUS_CONFIG" "Prometheus config" || true
    check_file_exists "$PROMETHEUS_ALERTS" "Prometheus alerts" || true
    check_file_exists "$ALERTMANAGER_CONFIG" "Alertmanager config" || true
    
    # Run validations based on mode
    if [[ "$YAML_ONLY" == "true" ]]; then
        run_yaml_validation
    elif [[ "$DOCKER_ONLY" == "true" ]]; then
        run_docker_validation
    else
        run_yaml_validation
        run_docker_validation
        run_common_checks
    fi
    
    # Summary
    print_header "Validation Summary"
    
    if [[ $VALIDATION_ERRORS -eq 0 ]]; then
        echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✓ All monitoring configuration validations passed!             ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
        exit 0
    else
        echo -e "\n${RED}╔══════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ✗ Validation failed with $VALIDATION_ERRORS error(s)                            ║${NC}"
        echo -e "${RED}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
        exit 1
    fi
}

# Run main function
main "$@"