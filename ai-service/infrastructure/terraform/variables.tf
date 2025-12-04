# =============================================================================
# Variables for RingRift AI Service Cloud Training Infrastructure
# =============================================================================

# -----------------------------------------------------------------------------
# General Settings
# -----------------------------------------------------------------------------

variable "project_name" {
  description = "Name of the project (used for resource naming)"
  type        = string
  default     = "ringrift-ai"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

# -----------------------------------------------------------------------------
# Networking
# -----------------------------------------------------------------------------

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 2

  validation {
    condition     = var.az_count >= 1 && var.az_count <= 3
    error_message = "AZ count must be between 1 and 3."
  }
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnet internet access"
  type        = bool
  default     = false
}

# -----------------------------------------------------------------------------
# ECS Configuration
# -----------------------------------------------------------------------------

variable "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  type        = string
  default     = "cmaes-training"
}

variable "worker_cpu" {
  description = "CPU units for worker tasks (1024 = 1 vCPU)"
  type        = number
  default     = 2048
}

variable "worker_memory" {
  description = "Memory in MiB for worker tasks"
  type        = number
  default     = 4096
}

variable "worker_desired_count" {
  description = "Desired number of worker tasks"
  type        = number
  default     = 4
}

variable "worker_min_count" {
  description = "Minimum number of worker tasks for auto-scaling"
  type        = number
  default     = 0
}

variable "worker_max_count" {
  description = "Maximum number of worker tasks for auto-scaling"
  type        = number
  default     = 20
}

variable "enable_spot_instances" {
  description = "Use Spot capacity for Fargate tasks"
  type        = bool
  default     = true
}

variable "spot_percentage" {
  description = "Percentage of tasks to run on Spot (0-100)"
  type        = number
  default     = 80

  validation {
    condition     = var.spot_percentage >= 0 && var.spot_percentage <= 100
    error_message = "Spot percentage must be between 0 and 100."
  }
}

# -----------------------------------------------------------------------------
# Container Configuration
# -----------------------------------------------------------------------------

variable "worker_image" {
  description = "Docker image for worker container"
  type        = string
  default     = ""
}

variable "ecr_repository_name" {
  description = "Name of ECR repository for worker images"
  type        = string
  default     = "ringrift-ai-worker"
}

variable "create_ecr_repository" {
  description = "Whether to create an ECR repository"
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# SQS Configuration
# -----------------------------------------------------------------------------

variable "sqs_visibility_timeout" {
  description = "SQS message visibility timeout in seconds"
  type        = number
  default     = 300

  validation {
    condition     = var.sqs_visibility_timeout >= 30 && var.sqs_visibility_timeout <= 43200
    error_message = "Visibility timeout must be between 30 and 43200 seconds."
  }
}

variable "sqs_message_retention" {
  description = "SQS message retention in seconds (max 14 days)"
  type        = number
  default     = 86400

  validation {
    condition     = var.sqs_message_retention >= 60 && var.sqs_message_retention <= 1209600
    error_message = "Message retention must be between 60 seconds and 14 days."
  }
}

variable "enable_dead_letter_queue" {
  description = "Enable dead letter queue for failed messages"
  type        = bool
  default     = true
}

variable "max_receive_count" {
  description = "Max receives before message goes to DLQ"
  type        = number
  default     = 3
}

# -----------------------------------------------------------------------------
# Redis Configuration (Optional)
# -----------------------------------------------------------------------------

variable "enable_redis" {
  description = "Enable ElastiCache Redis (alternative to SQS)"
  type        = bool
  default     = false
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.small"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 1
}

# -----------------------------------------------------------------------------
# S3 Configuration
# -----------------------------------------------------------------------------

variable "s3_bucket_name" {
  description = "S3 bucket name for state pools and checkpoints"
  type        = string
  default     = ""
}

variable "s3_force_destroy" {
  description = "Allow bucket destruction with objects (dev only)"
  type        = bool
  default     = false
}

variable "s3_versioning_enabled" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# Logging & Monitoring
# -----------------------------------------------------------------------------

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30

  validation {
    condition = contains([
      1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
    ], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch retention period."
  }
}

variable "enable_container_insights" {
  description = "Enable Container Insights for the ECS cluster"
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# CMA-ES Worker Configuration
# -----------------------------------------------------------------------------

variable "default_board_type" {
  description = "Default board type for workers"
  type        = string
  default     = "square8"
}

variable "default_num_players" {
  description = "Default number of players"
  type        = number
  default     = 2
}

variable "default_games_per_eval" {
  description = "Default games per evaluation"
  type        = number
  default     = 24
}

variable "default_state_pool_id" {
  description = "Default state pool ID"
  type        = string
  default     = "v1"
}

variable "preload_pools" {
  description = "Comma-separated pool specs to preload (e.g., 'square8_2p_v1,hex_3p_v1')"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# Tags
# -----------------------------------------------------------------------------

variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
