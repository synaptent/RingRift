# =============================================================================
# SQS Queues for CMA-ES Task Distribution
# =============================================================================

# -----------------------------------------------------------------------------
# Task Queue (Coordinator -> Workers)
# -----------------------------------------------------------------------------

resource "aws_sqs_queue" "tasks" {
  name = "${var.project_name}-${var.environment}-cmaes-tasks"

  visibility_timeout_seconds  = var.sqs_visibility_timeout
  message_retention_seconds   = var.sqs_message_retention
  delay_seconds               = 0
  receive_wait_time_seconds   = 20  # Long polling
  max_message_size            = 262144  # 256 KB

  # Enable dead letter queue if configured
  redrive_policy = var.enable_dead_letter_queue ? jsonencode({
    deadLetterTargetArn = aws_sqs_queue.tasks_dlq[0].arn
    maxReceiveCount     = var.max_receive_count
  }) : null

  tags = {
    Name    = "${var.project_name}-${var.environment}-cmaes-tasks"
    Purpose = "CMA-ES evaluation task distribution"
  }
}

resource "aws_sqs_queue" "tasks_dlq" {
  count = var.enable_dead_letter_queue ? 1 : 0
  name  = "${var.project_name}-${var.environment}-cmaes-tasks-dlq"

  message_retention_seconds = 1209600  # 14 days for DLQ

  tags = {
    Name    = "${var.project_name}-${var.environment}-cmaes-tasks-dlq"
    Purpose = "Dead letter queue for failed tasks"
  }
}

# -----------------------------------------------------------------------------
# Result Queue (Workers -> Coordinator)
# -----------------------------------------------------------------------------

resource "aws_sqs_queue" "results" {
  name = "${var.project_name}-${var.environment}-cmaes-results"

  visibility_timeout_seconds  = 30
  message_retention_seconds   = var.sqs_message_retention
  delay_seconds               = 0
  receive_wait_time_seconds   = 20  # Long polling
  max_message_size            = 262144  # 256 KB

  tags = {
    Name    = "${var.project_name}-${var.environment}-cmaes-results"
    Purpose = "CMA-ES evaluation results"
  }
}

# -----------------------------------------------------------------------------
# CloudWatch Alarms for Queue Monitoring
# -----------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "task_queue_depth" {
  alarm_name          = "${var.project_name}-${var.environment}-task-queue-depth"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 300
  statistic           = "Average"
  threshold           = 100
  alarm_description   = "Task queue depth is high - consider scaling workers"

  dimensions = {
    QueueName = aws_sqs_queue.tasks.name
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-task-queue-alarm"
  }
}

resource "aws_cloudwatch_metric_alarm" "dlq_messages" {
  count               = var.enable_dead_letter_queue ? 1 : 0
  alarm_name          = "${var.project_name}-${var.environment}-dlq-messages"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Messages in dead letter queue - investigate failures"

  dimensions = {
    QueueName = aws_sqs_queue.tasks_dlq[0].name
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-dlq-alarm"
  }
}
