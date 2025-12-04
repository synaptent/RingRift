# RingRift AI Service - Cloud Training Infrastructure

This directory contains Terraform configurations for deploying the CMA-ES distributed training infrastructure on AWS.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AWS Cloud                                      │
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │  Coordinator    │    │   SQS Queues    │    │   ECS Cluster   │    │
│  │  (Local/CI)     │───▶│  ┌───────────┐  │───▶│  ┌───────────┐  │    │
│  │                 │    │  │   Tasks   │  │    │  │  Worker 1 │  │    │
│  │  run_cmaes_     │    │  └───────────┘  │    │  └───────────┘  │    │
│  │  optimization.py│◀───│  ┌───────────┐  │◀───│  ┌───────────┐  │    │
│  │                 │    │  │  Results  │  │    │  │  Worker 2 │  │    │
│  └─────────────────┘    │  └───────────┘  │    │  └───────────┘  │    │
│                         └─────────────────┘    │       ...       │    │
│                                                │  ┌───────────┐  │    │
│  ┌─────────────────┐                          │  │  Worker N │  │    │
│  │   S3 Bucket     │                          │  └───────────┘  │    │
│  │  ┌───────────┐  │◀─────────────────────────┴─────────────────┘    │
│  │  │State Pools│  │                                                  │
│  │  └───────────┘  │                                                  │
│  │  ┌───────────┐  │                                                  │
│  │  │Checkpoints│  │                                                  │
│  │  └───────────┘  │                                                  │
│  └─────────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Terraform** >= 1.5
2. **AWS CLI** configured with appropriate credentials
3. **Docker** for building worker images

## Quick Start

### 1. Initialize Terraform

```bash
cd ai-service/infrastructure/terraform
terraform init
```

### 2. Deploy Development Environment

```bash
# Review the plan
terraform plan -var-file="environments/dev.tfvars"

# Apply the configuration
terraform apply -var-file="environments/dev.tfvars"
```

### 3. Build and Push Worker Image

```bash
# Get ECR login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(terraform output -raw ecr_repository_url)

# Build the image
cd ../..
docker build -t ringrift-ai-worker -f docker/Dockerfile.worker .

# Tag and push
ECR_URL=$(terraform output -raw ecr_repository_url)
docker tag ringrift-ai-worker:latest $ECR_URL:latest
docker push $ECR_URL:latest
```

### 4. Upload State Pools

```bash
BUCKET=$(terraform output -raw s3_bucket_name)
aws s3 sync data/eval_pools/ s3://$BUCKET/state-pools/
```

### 5. Run CMA-ES Optimization

```bash
# Export credentials
export AWS_ACCESS_KEY_ID=$(terraform output -raw coordinator_access_key_id)
export AWS_SECRET_ACCESS_KEY=$(terraform output -raw coordinator_secret_access_key)
export AWS_REGION=us-east-1

# Export queue URLs
export SQS_TASK_QUEUE_URL=$(terraform output -raw sqs_task_queue_url)
export SQS_RESULT_QUEUE_URL=$(terraform output -raw sqs_result_queue_url)

# Scale up workers
aws ecs update-service \
    --cluster $(terraform output -raw ecs_cluster_name) \
    --service $(terraform output -raw ecs_service_name) \
    --desired-count 10

# Run optimization
cd ../..
python scripts/run_cmaes_optimization.py \
    --generations 50 \
    --population-size 20 \
    --games-per-eval 30 \
    --board square8 \
    --num-players 2 \
    --queue-backend sqs \
    --output logs/cmaes/cloud_run

# Scale down when done
aws ecs update-service \
    --cluster $(terraform output -raw ecs_cluster_name) \
    --service $(terraform output -raw ecs_service_name) \
    --desired-count 0
```

## Configuration

### Environment Variables

| Variable                | Description                          | Default     |
| ----------------------- | ------------------------------------ | ----------- |
| `environment`           | Environment name (dev/staging/prod)  | `dev`       |
| `aws_region`            | AWS region                           | `us-east-1` |
| `worker_cpu`            | CPU units per worker (1024 = 1 vCPU) | `2048`      |
| `worker_memory`         | Memory per worker (MiB)              | `4096`      |
| `worker_desired_count`  | Initial number of workers            | `4`         |
| `enable_spot_instances` | Use Fargate Spot                     | `true`      |
| `spot_percentage`       | Percentage on Spot (0-100)           | `80`        |

See `variables.tf` for all available options.

### Cost Optimization

1. **Fargate Spot**: Workers use Spot instances by default (up to 70% savings)
2. **Auto-scaling**: Workers scale to zero when queue is empty
3. **S3 Lifecycle**: Old checkpoints automatically archived/deleted

### Estimated Costs (Dev Environment)

| Resource                      | Hourly Cost | Monthly (10h/day) |
| ----------------------------- | ----------- | ----------------- |
| 4x Fargate Spot (1 vCPU, 2GB) | ~$0.02      | ~$6               |
| SQS (1M requests)             | -           | ~$0.40            |
| S3 (10GB)                     | -           | ~$0.23            |
| NAT Gateway (disabled)        | $0          | $0                |
| **Total**                     |             | **~$7/month**     |

## Modules

| File           | Description                           |
| -------------- | ------------------------------------- |
| `main.tf`      | Provider config, VPC, networking      |
| `variables.tf` | Input variable definitions            |
| `outputs.tf`   | Output values                         |
| `sqs.tf`       | SQS task and result queues            |
| `ecs.tf`       | ECS cluster, task definition, service |
| `iam.tf`       | IAM roles and policies                |
| `s3.tf`        | S3 bucket for state pools             |
| `redis.tf`     | ElastiCache Redis (optional)          |

## Troubleshooting

### Workers Not Starting

```bash
# Check ECS service events
aws ecs describe-services \
    --cluster $(terraform output -raw ecs_cluster_name) \
    --services $(terraform output -raw ecs_service_name) \
    --query 'services[0].events[:5]'

# Check task logs
aws logs tail /ecs/ringrift-ai-dev/cmaes-worker --follow
```

### Tasks Not Being Processed

```bash
# Check queue depth
aws sqs get-queue-attributes \
    --queue-url $(terraform output -raw sqs_task_queue_url) \
    --attribute-names ApproximateNumberOfMessagesVisible

# Check dead letter queue
aws sqs get-queue-attributes \
    --queue-url $(terraform output -raw sqs_dlq_url) \
    --attribute-names ApproximateNumberOfMessagesVisible
```

### Force Worker Restart

```bash
aws ecs update-service \
    --cluster $(terraform output -raw ecs_cluster_name) \
    --service $(terraform output -raw ecs_service_name) \
    --force-new-deployment
```

## Cleanup

```bash
# Scale down workers first
aws ecs update-service \
    --cluster $(terraform output -raw ecs_cluster_name) \
    --service $(terraform output -raw ecs_service_name) \
    --desired-count 0

# Wait for tasks to drain, then destroy
terraform destroy -var-file="environments/dev.tfvars"
```

## Security Notes

1. **Credentials**: Store Terraform state remotely with encryption enabled
2. **IAM**: Coordinator user has minimal permissions
3. **Network**: Workers run in public subnets with security groups
4. **S3**: Bucket has public access blocked
5. **Secrets**: Use AWS Secrets Manager for production credentials
