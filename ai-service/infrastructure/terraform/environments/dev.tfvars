# =============================================================================
# Development Environment Configuration
# =============================================================================
# Apply with: terraform apply -var-file="environments/dev.tfvars"

environment = "dev"
aws_region  = "us-east-1"

# Networking - Minimal setup for dev
vpc_cidr           = "10.0.0.0/16"
az_count           = 2
enable_nat_gateway = false  # Save costs, use public subnets

# ECS - Small cluster for testing
ecs_cluster_name     = "cmaes-dev"
worker_cpu           = 1024  # 1 vCPU
worker_memory        = 2048  # 2 GB
worker_desired_count = 2
worker_min_count     = 0
worker_max_count     = 8
enable_spot_instances = true
spot_percentage       = 100  # All spot for dev

# ECR
create_ecr_repository = true
ecr_repository_name   = "ringrift-ai-worker-dev"

# SQS
sqs_visibility_timeout   = 300
sqs_message_retention    = 86400  # 1 day
enable_dead_letter_queue = true
max_receive_count        = 3

# Redis - Disabled for dev (use SQS only)
enable_redis = false

# S3
s3_force_destroy      = true  # Allow destroying bucket with objects
s3_versioning_enabled = false  # Save costs in dev

# Logging
log_retention_days        = 7
enable_container_insights = false  # Save costs in dev

# Worker defaults
default_board_type     = "square8"
default_num_players    = 2
default_games_per_eval = 24
default_state_pool_id  = "v1"
preload_pools          = "square8_2p_v1"

# Tags
additional_tags = {
  CostCenter = "development"
}
