#!/usr/bin/env python3
"""Configure S3 Lifecycle Policies for RingRift Storage Tier.

Phase 4 of S3-as-primary-storage: Auto-archive old data to reduce costs.

This script configures lifecycle policies on the ringrift-models-20251214 bucket:
- Games: STANDARD -> STANDARD_IA (30 days) -> GLACIER (180 days)
- Models: Stay in STANDARD (fast access)
- Training NPZ: STANDARD -> STANDARD_IA (30 days)

Usage:
    python scripts/configure_s3_lifecycle.py
    python scripts/configure_s3_lifecycle.py --dry-run
    python scripts/configure_s3_lifecycle.py --bucket my-custom-bucket

January 2026: Created as part of S3 first-class storage tier upgrade.
"""

import argparse
import json
import subprocess
import sys


def get_lifecycle_configuration() -> dict:
    """Get the lifecycle configuration for RingRift data.

    Returns:
        Dict containing lifecycle rules
    """
    return {
        "Rules": [
            {
                "ID": "archive-old-games",
                "Status": "Enabled",
                "Filter": {
                    "Prefix": "consolidated/games/"
                },
                "Transitions": [
                    {
                        "Days": 30,
                        "StorageClass": "STANDARD_IA"
                    },
                    {
                        "Days": 180,
                        "StorageClass": "GLACIER"
                    }
                ]
            },
            {
                "ID": "archive-old-training-npz",
                "Status": "Enabled",
                "Filter": {
                    "Prefix": "consolidated/training/"
                },
                "Transitions": [
                    {
                        "Days": 30,
                        "StorageClass": "STANDARD_IA"
                    }
                ]
            },
            {
                "ID": "keep-models-standard",
                "Status": "Enabled",
                "Filter": {
                    "Prefix": "consolidated/models/"
                },
                # No transitions - models stay in STANDARD for fast access
                "NoncurrentVersionExpiration": {
                    "NoncurrentDays": 90
                }
            },
            {
                "ID": "archive-node-data",
                "Status": "Enabled",
                "Filter": {
                    "Prefix": "nodes/"
                },
                "Transitions": [
                    {
                        "Days": 30,  # STANDARD_IA requires minimum 30 days
                        "StorageClass": "STANDARD_IA"
                    },
                    {
                        "Days": 90,
                        "StorageClass": "GLACIER"
                    }
                ]
            }
        ]
    }


def apply_lifecycle_configuration(bucket: str, dry_run: bool = False) -> bool:
    """Apply lifecycle configuration to S3 bucket.

    Args:
        bucket: S3 bucket name
        dry_run: If True, print configuration without applying

    Returns:
        True if successful
    """
    config = get_lifecycle_configuration()

    if dry_run:
        print(f"[DRY RUN] Would apply lifecycle configuration to s3://{bucket}/")
        print(json.dumps(config, indent=2))
        return True

    # Write config to temp file
    config_json = json.dumps(config)

    try:
        result = subprocess.run(
            [
                "aws", "s3api", "put-bucket-lifecycle-configuration",
                "--bucket", bucket,
                "--lifecycle-configuration", config_json,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print(f"Successfully applied lifecycle configuration to s3://{bucket}/")
            return True
        else:
            print(f"Error applying lifecycle configuration: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("Error: AWS CLI timed out")
        return False
    except FileNotFoundError:
        print("Error: AWS CLI not found. Please install aws-cli.")
        return False


def get_current_lifecycle_configuration(bucket: str) -> dict | None:
    """Get current lifecycle configuration from S3 bucket.

    Args:
        bucket: S3 bucket name

    Returns:
        Current configuration or None if not set
    """
    try:
        result = subprocess.run(
            [
                "aws", "s3api", "get-bucket-lifecycle-configuration",
                "--bucket", bucket,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            if "NoSuchLifecycleConfiguration" in result.stderr:
                return None
            print(f"Error getting lifecycle configuration: {result.stderr}")
            return None

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Configure S3 lifecycle policies")
    parser.add_argument(
        "--bucket",
        default="ringrift-models-20251214",
        help="S3 bucket name (default: ringrift-models-20251214)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without applying"
    )
    parser.add_argument(
        "--show-current",
        action="store_true",
        help="Show current lifecycle configuration"
    )

    args = parser.parse_args()

    if args.show_current:
        current = get_current_lifecycle_configuration(args.bucket)
        if current:
            print(f"Current lifecycle configuration for s3://{args.bucket}/:")
            print(json.dumps(current, indent=2))
        else:
            print(f"No lifecycle configuration set for s3://{args.bucket}/")
        return

    success = apply_lifecycle_configuration(args.bucket, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
