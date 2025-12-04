# =============================================================================
# ECS Cluster and Task Definitions for CMA-ES Workers
# =============================================================================

# -----------------------------------------------------------------------------
# ECS Cluster
# -----------------------------------------------------------------------------

resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-${var.environment}-${var.ecs_cluster_name}"

  setting {
    name  = "containerInsights"
    value = var.enable_container_insights ? "enabled" : "disabled"
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-${var.ecs_cluster_name}"
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name

  capacity_providers = var.enable_spot_instances ? ["FARGATE", "FARGATE_SPOT"] : ["FARGATE"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = var.enable_spot_instances ? var.spot_percentage : 0
    base              = 0
  }

  dynamic "default_capacity_provider_strategy" {
    for_each = var.enable_spot_instances ? [1] : []
    content {
      capacity_provider = "FARGATE"
      weight            = 100 - var.spot_percentage
      base              = 1
    }
  }
}

# -----------------------------------------------------------------------------
# ECR Repository (Optional)
# -----------------------------------------------------------------------------

resource "aws_ecr_repository" "worker" {
  count = var.create_ecr_repository ? 1 : 0
  name  = var.ecr_repository_name

  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name = var.ecr_repository_name
  }
}

resource "aws_ecr_lifecycle_policy" "worker" {
  count      = var.create_ecr_repository ? 1 : 0
  repository = aws_ecr_repository.worker[0].name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# CloudWatch Log Group
# -----------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/ecs/${var.project_name}-${var.environment}/cmaes-worker"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.project_name}-${var.environment}-cmaes-worker-logs"
  }
}

# -----------------------------------------------------------------------------
# ECS Task Definition
# -----------------------------------------------------------------------------

locals {
  worker_image = var.worker_image != "" ? var.worker_image : (
    var.create_ecr_repository ? "${aws_ecr_repository.worker[0].repository_url}:latest" : ""
  )
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${var.project_name}-${var.environment}-cmaes-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.worker_cpu
  memory                   = var.worker_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "cmaes-worker"
      image     = local.worker_image
      essential = true

      portMappings = [
        {
          containerPort = 8080
          hostPort      = 8080
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "QUEUE_BACKEND"
          value = "sqs"
        },
        {
          name  = "SQS_TASK_QUEUE_URL"
          value = aws_sqs_queue.tasks.url
        },
        {
          name  = "SQS_RESULT_QUEUE_URL"
          value = aws_sqs_queue.results.url
        },
        {
          name  = "SQS_REGION"
          value = var.aws_region
        },
        {
          name  = "STORAGE_BACKEND"
          value = "s3"
        },
        {
          name  = "STORAGE_BUCKET"
          value = aws_s3_bucket.storage.bucket
        },
        {
          name  = "PRELOAD_POOLS"
          value = var.preload_pools
        },
        {
          name  = "RINGRIFT_SKIP_SHADOW_CONTRACTS"
          value = "true"
        },
        {
          name  = "RINGRIFT_USE_MAKE_UNMAKE"
          value = "true"
        },
        {
          name  = "RINGRIFT_USE_BATCH_EVAL"
          value = "true"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.worker.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "worker"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }

      stopTimeout = 120  # Allow graceful shutdown
    }
  ])

  tags = {
    Name = "${var.project_name}-${var.environment}-cmaes-worker"
  }
}

# -----------------------------------------------------------------------------
# ECS Service
# -----------------------------------------------------------------------------

resource "aws_ecs_service" "worker" {
  name            = "${var.project_name}-${var.environment}-cmaes-worker"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = var.worker_desired_count

  launch_type = null  # Use capacity provider strategy instead

  capacity_provider_strategy {
    capacity_provider = var.enable_spot_instances ? "FARGATE_SPOT" : "FARGATE"
    weight            = var.enable_spot_instances ? var.spot_percentage : 100
    base              = 0
  }

  dynamic "capacity_provider_strategy" {
    for_each = var.enable_spot_instances ? [1] : []
    content {
      capacity_provider = "FARGATE"
      weight            = 100 - var.spot_percentage
      base              = 1
    }
  }

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true
  }

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 50
  }

  # Allow graceful shutdown for Spot termination
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-cmaes-worker"
  }

  lifecycle {
    ignore_changes = [desired_count]  # Managed by auto-scaling
  }
}

# -----------------------------------------------------------------------------
# Auto Scaling
# -----------------------------------------------------------------------------

resource "aws_appautoscaling_target" "worker" {
  max_capacity       = var.worker_max_count
  min_capacity       = var.worker_min_count
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.worker.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# Scale based on SQS queue depth
resource "aws_appautoscaling_policy" "worker_queue_depth" {
  name               = "${var.project_name}-${var.environment}-worker-queue-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  service_namespace  = aws_appautoscaling_target.worker.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = 10  # Target 10 messages per worker
    scale_in_cooldown  = 300
    scale_out_cooldown = 60

    customized_metric_specification {
      metric_name = "BacklogPerTask"
      namespace   = "RingRift/CMAES"
      statistic   = "Average"
      unit        = "Count"

      dimensions {
        name  = "QueueName"
        value = aws_sqs_queue.tasks.name
      }

      dimensions {
        name  = "ServiceName"
        value = aws_ecs_service.worker.name
      }
    }
  }
}

# Scale to zero when queue is empty (cost savings)
resource "aws_appautoscaling_policy" "worker_scale_to_zero" {
  name               = "${var.project_name}-${var.environment}-worker-scale-to-zero"
  policy_type        = "StepScaling"
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  service_namespace  = aws_appautoscaling_target.worker.service_namespace

  step_scaling_policy_configuration {
    adjustment_type         = "ExactCapacity"
    cooldown                = 300
    metric_aggregation_type = "Average"

    step_adjustment {
      metric_interval_upper_bound = 0
      scaling_adjustment          = 0
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "queue_empty" {
  alarm_name          = "${var.project_name}-${var.environment}-queue-empty"
  comparison_operator = "LessThanOrEqualToThreshold"
  evaluation_periods  = 5
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 60
  statistic           = "Average"
  threshold           = 0
  alarm_description   = "Scale to zero when queue is empty"

  dimensions = {
    QueueName = aws_sqs_queue.tasks.name
  }

  alarm_actions = [aws_appautoscaling_policy.worker_scale_to_zero.arn]

  tags = {
    Name = "${var.project_name}-${var.environment}-queue-empty-alarm"
  }
}
