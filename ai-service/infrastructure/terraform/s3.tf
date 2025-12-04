# =============================================================================
# S3 Bucket for State Pools, Checkpoints, and Training Data
# =============================================================================

locals {
  bucket_name = var.s3_bucket_name != "" ? var.s3_bucket_name : "${var.project_name}-${var.environment}-training-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "storage" {
  bucket        = local.bucket_name
  force_destroy = var.s3_force_destroy

  tags = {
    Name    = local.bucket_name
    Purpose = "CMA-ES training data and checkpoints"
  }
}

resource "aws_s3_bucket_versioning" "storage" {
  bucket = aws_s3_bucket.storage.id

  versioning_configuration {
    status = var.s3_versioning_enabled ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "storage" {
  bucket = aws_s3_bucket.storage.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "storage" {
  bucket = aws_s3_bucket.storage.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "storage" {
  bucket = aws_s3_bucket.storage.id

  rule {
    id     = "cleanup-old-checkpoints"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    # Move to infrequent access after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    # Delete after 90 days
    expiration {
      days = 90
    }

    # Clean up old versions
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }

  rule {
    id     = "cleanup-old-logs"
    status = "Enabled"

    filter {
      prefix = "logs/"
    }

    expiration {
      days = 30
    }
  }

  rule {
    id     = "keep-state-pools"
    status = "Enabled"

    filter {
      prefix = "state-pools/"
    }

    # State pools are kept indefinitely but moved to IA
    transition {
      days          = 60
      storage_class = "STANDARD_IA"
    }
  }

  rule {
    id     = "keep-trained-weights"
    status = "Enabled"

    filter {
      prefix = "trained-weights/"
    }

    # Trained weights moved to IA but never deleted
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 365
      storage_class = "GLACIER"
    }
  }
}

# -----------------------------------------------------------------------------
# S3 Bucket for Terraform State (Optional)
# -----------------------------------------------------------------------------

# Uncomment to create a bucket for Terraform remote state
# resource "aws_s3_bucket" "terraform_state" {
#   bucket = "ringrift-terraform-state"
#
#   tags = {
#     Name    = "ringrift-terraform-state"
#     Purpose = "Terraform state storage"
#   }
# }
#
# resource "aws_s3_bucket_versioning" "terraform_state" {
#   bucket = aws_s3_bucket.terraform_state.id
#
#   versioning_configuration {
#     status = "Enabled"
#   }
# }
#
# resource "aws_dynamodb_table" "terraform_locks" {
#   name         = "terraform-state-lock"
#   billing_mode = "PAY_PER_REQUEST"
#   hash_key     = "LockID"
#
#   attribute {
#     name = "LockID"
#     type = "S"
#   }
#
#   tags = {
#     Name    = "terraform-state-lock"
#     Purpose = "Terraform state locking"
#   }
# }
