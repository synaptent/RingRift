# =============================================================================
# Terraform Outputs
# =============================================================================

# -----------------------------------------------------------------------------
# VPC and Networking
# -----------------------------------------------------------------------------

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = aws_subnet.private[*].id
}

# -----------------------------------------------------------------------------
# ECS
# -----------------------------------------------------------------------------

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.worker.name
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = var.create_ecr_repository ? aws_ecr_repository.worker[0].repository_url : ""
}

# -----------------------------------------------------------------------------
# SQS Queues
# -----------------------------------------------------------------------------

output "sqs_task_queue_url" {
  description = "URL of the SQS task queue"
  value       = aws_sqs_queue.tasks.url
}

output "sqs_task_queue_arn" {
  description = "ARN of the SQS task queue"
  value       = aws_sqs_queue.tasks.arn
}

output "sqs_result_queue_url" {
  description = "URL of the SQS result queue"
  value       = aws_sqs_queue.results.url
}

output "sqs_result_queue_arn" {
  description = "ARN of the SQS result queue"
  value       = aws_sqs_queue.results.arn
}

output "sqs_dlq_url" {
  description = "URL of the dead letter queue"
  value       = var.enable_dead_letter_queue ? aws_sqs_queue.tasks_dlq[0].url : ""
}

# -----------------------------------------------------------------------------
# S3
# -----------------------------------------------------------------------------

output "s3_bucket_name" {
  description = "Name of the S3 storage bucket"
  value       = aws_s3_bucket.storage.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 storage bucket"
  value       = aws_s3_bucket.storage.arn
}

# -----------------------------------------------------------------------------
# Redis (if enabled)
# -----------------------------------------------------------------------------

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = var.enable_redis ? aws_elasticache_cluster.redis[0].cache_nodes[0].address : ""
}

output "redis_port" {
  description = "Redis port"
  value       = var.enable_redis ? aws_elasticache_cluster.redis[0].cache_nodes[0].port : 6379
}

# -----------------------------------------------------------------------------
# IAM
# -----------------------------------------------------------------------------

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role"
  value       = aws_iam_role.ecs_task.arn
}

output "coordinator_access_key_id" {
  description = "Access key ID for coordinator user"
  value       = aws_iam_access_key.coordinator.id
  sensitive   = true
}

output "coordinator_secret_access_key" {
  description = "Secret access key for coordinator user"
  value       = aws_iam_access_key.coordinator.secret
  sensitive   = true
}

# -----------------------------------------------------------------------------
# Environment Variables for Coordinator
# -----------------------------------------------------------------------------

output "coordinator_env_vars" {
  description = "Environment variables for running the coordinator"
  value = {
    QUEUE_BACKEND        = "sqs"
    SQS_TASK_QUEUE_URL   = aws_sqs_queue.tasks.url
    SQS_RESULT_QUEUE_URL = aws_sqs_queue.results.url
    SQS_REGION           = var.aws_region
    STORAGE_BACKEND      = "s3"
    STORAGE_BUCKET       = aws_s3_bucket.storage.bucket
    AWS_REGION           = var.aws_region
  }
}

# -----------------------------------------------------------------------------
# Quick Start Commands
# -----------------------------------------------------------------------------

output "quick_start" {
  description = "Quick start commands for using the infrastructure"
  value       = <<-EOT

    # ============================================================
    # RingRift AI Service - Cloud Training Infrastructure
    # ============================================================

    # 1. Build and push the worker Docker image:
    aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${var.create_ecr_repository ? aws_ecr_repository.worker[0].repository_url : "<ECR_URL>"}
    docker build -t ${var.ecr_repository_name} -f docker/Dockerfile.worker .
    docker tag ${var.ecr_repository_name}:latest ${var.create_ecr_repository ? aws_ecr_repository.worker[0].repository_url : "<ECR_URL>"}:latest
    docker push ${var.create_ecr_repository ? aws_ecr_repository.worker[0].repository_url : "<ECR_URL>"}:latest

    # 2. Upload state pools to S3:
    aws s3 sync data/eval_pools/ s3://${aws_s3_bucket.storage.bucket}/state-pools/

    # 3. Scale up workers:
    aws ecs update-service --cluster ${aws_ecs_cluster.main.name} --service ${aws_ecs_service.worker.name} --desired-count ${var.worker_desired_count}

    # 4. Run CMA-ES optimization with cloud workers:
    export AWS_ACCESS_KEY_ID=$(terraform output -raw coordinator_access_key_id)
    export AWS_SECRET_ACCESS_KEY=$(terraform output -raw coordinator_secret_access_key)
    export AWS_REGION=${var.aws_region}
    export SQS_TASK_QUEUE_URL=${aws_sqs_queue.tasks.url}
    export SQS_RESULT_QUEUE_URL=${aws_sqs_queue.results.url}

    python scripts/run_cmaes_optimization.py \
        --generations 50 \
        --population-size 20 \
        --games-per-eval 30 \
        --board square8 \
        --num-players 2 \
        --queue-backend sqs \
        --output logs/cmaes/cloud_run

    # 5. Scale down workers when done:
    aws ecs update-service --cluster ${aws_ecs_cluster.main.name} --service ${aws_ecs_service.worker.name} --desired-count 0

    EOT
}
