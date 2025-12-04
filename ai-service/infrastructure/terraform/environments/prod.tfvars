# =============================================================================
# Production Environment Configuration
# =============================================================================
# Apply with: terraform apply -var-file="environments/prod.tfvars"

environment = "prod"
aws_region  = "us-east-1"

# Networking - Full setup with private subnets
vpc_cidr           = "10.0.0.0/16"
az_count           = 3
enable_nat_gateway = true  # Required for private subnets

# ECS - Larger cluster for production training
ecs_cluster_name     = "cmaes-prod"
worker_cpu           = 4096  # 4 vCPUs
worker_memory        = 8192  # 8 GB
worker_desired_count = 10
worker_min_count     = 2
worker_max_count     = 50
enable_spot_instances = true
spot_percentage       = 80  # Mix of Spot and On-Demand

# ECR
create_ecr_repository = true
ecr_repository_name   = "ringrift-ai-worker"

# SQS
sqs_visibility_timeout   = 600   # Longer for complex evaluations
sqs_message_retention    = 345600  # 4 days
enable_dead_letter_queue = true
max_receive_count        = 5

# Redis - Optional, enable for lower latency
enable_redis          = false
redis_node_type       = "cache.r6g.large"
redis_num_cache_nodes = 2

# S3
s3_force_destroy      = false  # Protect production data
s3_versioning_enabled = true

# Logging
log_retention_days        = 90
enable_container_insights = true

# Worker defaults
default_board_type     = "square8"
default_num_players    = 2
default_games_per_eval = 30
default_state_pool_id  = "v1"
preload_pools          = "square8_2p_v1,square8_3p_v1,square8_4p_v1,square19_2p_v1,hex_2p_v1"

# Tags
additional_tags = {
  CostCenter = "ai-training"
  DataClass  = "internal"
}
