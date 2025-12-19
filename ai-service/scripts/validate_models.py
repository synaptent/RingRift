#!/usr/bin/env python3
"""
Model Validation and Cleanup Script

This script validates PyTorch model files and removes corrupted ones.
It can also update the ELO database to remove references to invalid models.

Usage:
    # Scan and report on model health
    python scripts/validate_models.py --scan

    # Remove corrupted models (dry-run first)
    python scripts/validate_models.py --cleanup --dry-run

    # Actually remove corrupted models
    python scripts/validate_models.py --cleanup

    # Also clean up ELO database entries for missing models
    python scripts/validate_models.py --cleanup --update-db

    # Run on remote host
    python scripts/validate_models.py --scan --host vast-rtx3060
"""

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("validate_models")

AI_SERVICE_ROOT = Path(__file__).parent.parent
MODELS_DIR = AI_SERVICE_ROOT / "models"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"


@dataclass
class ModelValidationResult:
    """Result of validating a single model file."""
    path: Path
    valid: bool
    size_bytes: int
    error: Optional[str] = None
    load_time_ms: Optional[float] = None
    checksum: Optional[str] = None


@dataclass
class ValidationReport:
    """Summary report of model validation."""
    total_models: int = 0
    valid_models: int = 0
    corrupted_models: int = 0
    missing_models: int = 0
    zero_byte_models: int = 0
    total_size_bytes: int = 0
    corrupted_size_bytes: int = 0
    results: List[ModelValidationResult] = field(default_factory=list)

    def add_result(self, result: ModelValidationResult):
        self.results.append(result)
        self.total_models += 1
        self.total_size_bytes += result.size_bytes

        if result.valid:
            self.valid_models += 1
        else:
            self.corrupted_models += 1
            self.corrupted_size_bytes += result.size_bytes
            if result.size_bytes == 0:
                self.zero_byte_models += 1

    def get_corrupted_paths(self) -> List[Path]:
        return [r.path for r in self.results if not r.valid]

    def print_summary(self):
        print("\n" + "=" * 70)
        print(" Model Validation Report")
        print("=" * 70)
        print(f"  Total models scanned:  {self.total_models}")
        print(f"  Valid models:          {self.valid_models} ({self.valid_models / max(1, self.total_models) * 100:.1f}%)")
        print(f"  Corrupted models:      {self.corrupted_models}")
        print(f"  Zero-byte models:      {self.zero_byte_models}")
        print(f"  Total size:            {self.total_size_bytes / 1024 / 1024:.1f} MB")
        print(f"  Corrupted size:        {self.corrupted_size_bytes / 1024 / 1024:.1f} MB")
        print("=" * 70)

        if self.corrupted_models > 0:
            print("\nCorrupted models:")
            for result in self.results:
                if not result.valid:
                    size_str = f"{result.size_bytes / 1024 / 1024:.1f}MB" if result.size_bytes > 0 else "0B"
                    print(f"  [{size_str}] {result.path.name}")
                    if result.error:
                        print(f"           Error: {result.error[:60]}...")


def validate_model_file(model_path: Path, compute_checksum: bool = False) -> ModelValidationResult:
    """
    Validate a single model file by attempting to load it.

    Args:
        model_path: Path to the .pth file
        compute_checksum: Whether to compute MD5 checksum (slower)

    Returns:
        ModelValidationResult with validation status
    """
    if not model_path.exists():
        return ModelValidationResult(
            path=model_path,
            valid=False,
            size_bytes=0,
            error="File does not exist"
        )

    size_bytes = model_path.stat().st_size

    # Zero-byte files are definitely corrupt
    if size_bytes == 0:
        return ModelValidationResult(
            path=model_path,
            valid=False,
            size_bytes=0,
            error="Zero-byte file"
        )

    # Very small files (< 1KB) are suspicious
    if size_bytes < 1024:
        return ModelValidationResult(
            path=model_path,
            valid=False,
            size_bytes=size_bytes,
            error=f"Suspiciously small file ({size_bytes} bytes)"
        )

    # Try to load the model
    start_time = time.time()
    try:
        import torch
        # Use map_location to avoid GPU memory issues during validation
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        load_time_ms = (time.time() - start_time) * 1000

        # Basic sanity checks on loaded checkpoint
        if checkpoint is None:
            return ModelValidationResult(
                path=model_path,
                valid=False,
                size_bytes=size_bytes,
                error="Loaded checkpoint is None"
            )

        # Compute checksum if requested
        checksum = None
        if compute_checksum:
            with open(model_path, 'rb') as f:
                checksum = hashlib.md5(f.read()).hexdigest()

        return ModelValidationResult(
            path=model_path,
            valid=True,
            size_bytes=size_bytes,
            load_time_ms=load_time_ms,
            checksum=checksum
        )

    except Exception as e:
        error_msg = str(e)
        # Common corruption errors
        if "zip archive" in error_msg.lower():
            error_msg = "Corrupted zip archive (incomplete transfer?)"
        elif "central directory" in error_msg.lower():
            error_msg = "Missing central directory (truncated file)"
        elif "unexpected end" in error_msg.lower():
            error_msg = "Unexpected end of file (incomplete write)"

        return ModelValidationResult(
            path=model_path,
            valid=False,
            size_bytes=size_bytes,
            error=error_msg
        )


def scan_models(
    models_dir: Path,
    pattern: str = "*.pth",
    parallel: bool = True,
    max_workers: int = 4
) -> ValidationReport:
    """
    Scan all model files in a directory and validate them.

    Args:
        models_dir: Directory containing model files
        pattern: Glob pattern for model files
        parallel: Whether to validate in parallel
        max_workers: Number of parallel workers

    Returns:
        ValidationReport with results
    """
    report = ValidationReport()
    model_files = list(models_dir.glob(pattern))

    if not model_files:
        logger.warning(f"No model files found matching {pattern} in {models_dir}")
        return report

    logger.info(f"Scanning {len(model_files)} model files...")

    if parallel and len(model_files) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(validate_model_file, path): path
                for path in model_files
            }
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                report.add_result(result)
                if (i + 1) % 10 == 0:
                    logger.info(f"  Progress: {i + 1}/{len(model_files)}")
    else:
        for i, path in enumerate(model_files):
            result = validate_model_file(path)
            report.add_result(result)
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{len(model_files)}")

    return report


def cleanup_corrupted_models(
    report: ValidationReport,
    dry_run: bool = True,
    backup_dir: Optional[Path] = None
) -> List[Path]:
    """
    Remove corrupted model files.

    Args:
        report: ValidationReport from scan_models
        dry_run: If True, only report what would be deleted
        backup_dir: If provided, move files here instead of deleting

    Returns:
        List of paths that were (or would be) removed
    """
    corrupted_paths = report.get_corrupted_paths()

    if not corrupted_paths:
        logger.info("No corrupted models to clean up")
        return []

    logger.info(f"{'Would remove' if dry_run else 'Removing'} {len(corrupted_paths)} corrupted models...")

    removed = []
    for path in corrupted_paths:
        if dry_run:
            logger.info(f"  [DRY-RUN] Would remove: {path.name}")
        else:
            try:
                if backup_dir:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = backup_dir / path.name
                    path.rename(backup_path)
                    logger.info(f"  Moved to backup: {path.name}")
                else:
                    path.unlink()
                    logger.info(f"  Removed: {path.name}")
                removed.append(path)
            except Exception as e:
                logger.error(f"  Failed to remove {path.name}: {e}")

    return removed


def update_elo_database(
    db_path: Path,
    removed_models: List[Path],
    dry_run: bool = True
) -> int:
    """
    Remove ELO database entries for models that no longer exist.

    Args:
        db_path: Path to unified_elo.db
        removed_models: List of model paths that were removed
        dry_run: If True, only report what would be changed

    Returns:
        Number of entries removed
    """
    if not db_path.exists():
        logger.warning(f"ELO database not found: {db_path}")
        return 0

    # Extract model IDs from paths (filename without extension)
    removed_ids = {p.stem for p in removed_models}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find entries in elo_ratings that match removed models
    cursor.execute("SELECT participant_id FROM elo_ratings")
    all_participants = {row[0] for row in cursor.fetchall()}

    to_remove = all_participants & removed_ids

    if not to_remove:
        logger.info("No ELO database entries to clean up")
        conn.close()
        return 0

    logger.info(f"{'Would remove' if dry_run else 'Removing'} {len(to_remove)} ELO database entries...")

    if not dry_run:
        for model_id in to_remove:
            cursor.execute("DELETE FROM elo_ratings WHERE participant_id = ?", (model_id,))
            cursor.execute("DELETE FROM match_history WHERE model_a_id = ? OR model_b_id = ?",
                          (model_id, model_id))
            cursor.execute("DELETE FROM rating_history WHERE participant_id = ?", (model_id,))
        conn.commit()

    conn.close()
    return len(to_remove)


def validate_model_quick(model_path: Path) -> bool:
    """
    Quick validation of a model file without full load.
    Checks file size and zip header.

    This is suitable for use in training loops where speed matters.
    """
    if not model_path.exists():
        return False

    size = model_path.stat().st_size
    if size < 1024:  # Less than 1KB is definitely wrong
        return False

    # Check for valid zip header (PyTorch saves as zip)
    try:
        with open(model_path, 'rb') as f:
            header = f.read(4)
            # PK zip header
            if header[:2] != b'PK':
                return False
    except Exception:
        return False

    return True


def validate_model_after_save(model_path: Path, expected_keys: Optional[List[str]] = None) -> bool:
    """
    Validate a model immediately after saving.
    This should be called after torch.save() to ensure the file is valid.

    Args:
        model_path: Path to the saved model
        expected_keys: Optional list of keys that should be in the checkpoint

    Returns:
        True if model is valid, False otherwise
    """
    try:
        import torch

        # First do quick validation
        if not validate_model_quick(model_path):
            return False

        # Load and verify
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        if checkpoint is None:
            return False

        # Check expected keys if provided
        if expected_keys:
            if isinstance(checkpoint, dict):
                for key in expected_keys:
                    if key not in checkpoint:
                        logger.warning(f"Missing expected key '{key}' in checkpoint")
                        return False

        return True

    except Exception as e:
        logger.error(f"Model validation failed for {model_path}: {e}")
        return False


def safe_model_save(
    checkpoint: dict,
    save_path: Path,
    validate: bool = True,
    backup_on_fail: bool = True
) -> bool:
    """
    Safely save a model checkpoint with validation.

    This function:
    1. Saves to a temporary file first
    2. Validates the saved file
    3. Atomically moves to final destination
    4. Optionally backs up on failure

    Args:
        checkpoint: The checkpoint dict to save
        save_path: Final destination path
        validate: Whether to validate after save
        backup_on_fail: Whether to keep temp file on failure

    Returns:
        True if save was successful, False otherwise
    """
    import torch

    save_path = Path(save_path)
    temp_path = save_path.with_suffix('.pth.tmp')

    try:
        # Save to temporary file
        torch.save(checkpoint, temp_path)

        # Ensure data is flushed to disk
        # This is important for NFS and network filesystems
        os.sync()

        # Validate if requested
        if validate:
            if not validate_model_after_save(temp_path):
                logger.error(f"Validation failed for {save_path}")
                if not backup_on_fail:
                    temp_path.unlink(missing_ok=True)
                return False

        # Atomic move to final destination
        temp_path.rename(save_path)

        logger.debug(f"Successfully saved and validated: {save_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save model {save_path}: {e}")
        if not backup_on_fail:
            temp_path.unlink(missing_ok=True)
        return False


def run_remote_validation(host: str, models_dir: str = "/root/ringrift/ai-service/models") -> Optional[ValidationReport]:
    """
    Run model validation on a remote host via SSH.

    Args:
        host: SSH host identifier (from distributed_hosts.yaml)
        models_dir: Remote models directory

    Returns:
        ValidationReport or None if connection failed
    """
    # Load host config
    hosts_config_path = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"
    if not hosts_config_path.exists():
        logger.error(f"Hosts config not found: {hosts_config_path}")
        return None

    import yaml
    with open(hosts_config_path) as f:
        hosts = yaml.safe_load(f)

    if host not in hosts:
        logger.error(f"Unknown host: {host}")
        return None

    host_config = hosts[host]
    ssh_host = host_config.get('ssh_host', host_config.get('host'))
    ssh_port = host_config.get('ssh_port', 22)

    # Build validation script to run remotely
    remote_script = f'''
import sys
import json
from pathlib import Path

sys.path.insert(0, '/root/ringrift/ai-service')

results = []
models_dir = Path("{models_dir}")

for model_path in models_dir.glob("*.pth"):
    result = {{"path": str(model_path), "size": model_path.stat().st_size}}

    if result["size"] == 0:
        result["valid"] = False
        result["error"] = "Zero-byte file"
    elif result["size"] < 1024:
        result["valid"] = False
        result["error"] = "Suspiciously small"
    else:
        try:
            import torch
            torch.load(model_path, map_location='cpu', weights_only=False)
            result["valid"] = True
        except Exception as e:
            result["valid"] = False
            result["error"] = str(e)[:100]

    results.append(result)

print(json.dumps(results))
'''

    try:
        cmd = [
            "ssh", "-o", "ConnectTimeout=30", "-p", str(ssh_port),
            f"root@{ssh_host}",
            f"python3 -c '{remote_script}'"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error(f"Remote validation failed: {result.stderr}")
            return None

        # Parse results
        remote_results = json.loads(result.stdout.strip().split('\n')[-1])

        report = ValidationReport()
        for r in remote_results:
            report.add_result(ModelValidationResult(
                path=Path(r["path"]),
                valid=r["valid"],
                size_bytes=r["size"],
                error=r.get("error")
            ))

        return report

    except subprocess.TimeoutExpired:
        logger.error(f"Remote validation timed out for {host}")
        return None
    except Exception as e:
        logger.error(f"Remote validation error for {host}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Validate and clean up model files")
    parser.add_argument("--scan", action="store_true", help="Scan models and report status")
    parser.add_argument("--cleanup", action="store_true", help="Remove corrupted models")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--update-db", action="store_true", help="Also update ELO database")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Models directory")
    parser.add_argument("--pattern", default="*.pth", help="File pattern to match")
    parser.add_argument("--host", help="Run on remote host instead of local")
    parser.add_argument("--backup-dir", type=Path, help="Move corrupted files here instead of deleting")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.scan and not args.cleanup:
        args.scan = True  # Default to scan

    # Run on remote host if specified
    if args.host:
        logger.info(f"Running validation on remote host: {args.host}")
        report = run_remote_validation(args.host)
        if report:
            report.print_summary()
        return

    # Local validation
    if not args.models_dir.exists():
        logger.error(f"Models directory not found: {args.models_dir}")
        sys.exit(1)

    # Scan models
    report = scan_models(args.models_dir, args.pattern)
    report.print_summary()

    # Cleanup if requested
    if args.cleanup and report.corrupted_models > 0:
        print()
        removed = cleanup_corrupted_models(
            report,
            dry_run=args.dry_run,
            backup_dir=args.backup_dir
        )

        if args.update_db and removed and not args.dry_run:
            update_elo_database(ELO_DB_PATH, removed, dry_run=args.dry_run)

    # Exit with error code if there were corrupted models
    if report.corrupted_models > 0:
        sys.exit(1)


# =============================================================================
# Training Loop Integration
# =============================================================================


class ModelHygieneChecker:
    """
    Model hygiene checker for integration into training loops.

    Usage in training scripts:
        from scripts.validate_models import ModelHygieneChecker

        # Create checker (runs validation every 30 minutes by default)
        hygiene = ModelHygieneChecker(models_dir=Path("models"), interval_minutes=30)

        # In training loop:
        for epoch in range(epochs):
            train_epoch()
            hygiene.check_and_cleanup()  # Only runs if interval has passed
    """

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        interval_minutes: int = 30,
        auto_cleanup: bool = True,
        log_only: bool = False,
    ):
        """
        Initialize the hygiene checker.

        Args:
            models_dir: Directory containing model files
            interval_minutes: How often to run validation (default 30 min)
            auto_cleanup: Whether to automatically remove corrupted models
            log_only: If True, only log issues without removing files
        """
        self.models_dir = models_dir
        self.interval_seconds = interval_minutes * 60
        self.auto_cleanup = auto_cleanup
        self.log_only = log_only
        self.last_check_time = 0.0
        self.corrupted_count = 0
        self.cleaned_count = 0

    def check_and_cleanup(self, force: bool = False) -> Dict[str, int]:
        """
        Run validation if enough time has passed since last check.

        Args:
            force: If True, run regardless of interval

        Returns:
            Dict with 'scanned', 'valid', 'corrupted', 'cleaned' counts
        """
        current_time = time.time()

        if not force and (current_time - self.last_check_time) < self.interval_seconds:
            return {"skipped": True}

        self.last_check_time = current_time

        # Run scan
        report = scan_models(self.models_dir, parallel=True, max_workers=2)

        result = {
            "scanned": report.total_models,
            "valid": report.valid_models,
            "corrupted": report.corrupted_models,
            "cleaned": 0,
        }

        self.corrupted_count += report.corrupted_models

        if report.corrupted_models > 0:
            if self.log_only:
                logger.warning(
                    f"[ModelHygiene] Found {report.corrupted_models} corrupted models "
                    f"(log_only mode, not cleaning)"
                )
                for r in report.results:
                    if not r.valid:
                        logger.warning(f"  Corrupted: {r.path.name} - {r.error}")
            elif self.auto_cleanup:
                removed = cleanup_corrupted_models(report, dry_run=False)
                result["cleaned"] = len(removed)
                self.cleaned_count += len(removed)
                logger.info(
                    f"[ModelHygiene] Cleaned {len(removed)} corrupted models"
                )

        return result

    def get_stats(self) -> Dict[str, int]:
        """Get cumulative statistics."""
        return {
            "total_corrupted_found": self.corrupted_count,
            "total_cleaned": self.cleaned_count,
            "checks_performed": int(self.last_check_time > 0),
        }


def validate_checkpoint_after_save(checkpoint_path: Path, expected_keys: Optional[List[str]] = None) -> bool:
    """
    Quick validation after saving a checkpoint.
    Call this immediately after torch.save() to verify the file is valid.

    Args:
        checkpoint_path: Path to the saved checkpoint
        expected_keys: Optional list of keys that should be in the checkpoint

    Returns:
        True if valid, False if corrupted
    """
    return validate_model_after_save(checkpoint_path, expected_keys)


def run_startup_validation(models_dir: Path = MODELS_DIR, cleanup: bool = True) -> ValidationReport:
    """
    Run model validation at training startup.
    Recommended to call this at the beginning of training scripts.

    Args:
        models_dir: Directory containing model files
        cleanup: Whether to remove corrupted models

    Returns:
        ValidationReport with results
    """
    logger.info(f"[Startup Validation] Scanning models in {models_dir}...")

    report = scan_models(models_dir, parallel=True)

    if report.corrupted_models > 0:
        logger.warning(
            f"[Startup Validation] Found {report.corrupted_models} corrupted models "
            f"({report.corrupted_size_bytes / 1024 / 1024:.1f} MB)"
        )

        if cleanup:
            removed = cleanup_corrupted_models(report, dry_run=False)
            logger.info(f"[Startup Validation] Removed {len(removed)} corrupted models")
    else:
        logger.info(
            f"[Startup Validation] All {report.valid_models} models are valid"
        )

    return report


if __name__ == "__main__":
    main()
