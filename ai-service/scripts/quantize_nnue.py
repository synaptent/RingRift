#!/usr/bin/env python3
"""Quantize NNUE models to int8 for faster CPU inference.

This script converts float32 NNUE models to int8 quantized versions,
which can provide 2-4x speedup on CPU inference with minimal accuracy loss.

Usage:
    # Quantize a single model
    python scripts/quantize_nnue.py models/nnue_square8_best.pth

    # Quantize all NNUE models in a directory
    python scripts/quantize_nnue.py --all models/

    # Benchmark quantization speedup
    python scripts/quantize_nnue.py --benchmark models/nnue_square8_best.pth
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

import numpy as np
import torch

from app.ai.nnue import RingRiftNNUE, get_feature_dim
from app.models import BoardType


def parse_board_type(model_path: str) -> BoardType:
    """Infer board type from model path."""
    path_lower = model_path.lower()
    if "square19" in path_lower or "sq19" in path_lower:
        return BoardType.SQUARE19
    elif "hex" in path_lower:
        return BoardType.HEXAGONAL
    return BoardType.SQUARE8


def load_model(model_path: str) -> RingRiftNNUE:
    """Load an NNUE model from checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu")

    # Get board type from checkpoint or path
    if "board_type" in checkpoint:
        board_type = BoardType(checkpoint["board_type"])
    else:
        board_type = parse_board_type(model_path)

    hidden_dim = checkpoint.get("hidden_dim", 256)
    num_hidden_layers = checkpoint.get("num_hidden_layers", 2)

    model = RingRiftNNUE(
        board_type=board_type,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def quantize_model(model: RingRiftNNUE) -> torch.nn.Module:
    """Apply dynamic int8 quantization to the model."""
    model.cpu()
    model.eval()

    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    return quantized


def save_quantized_model(
    original_checkpoint_path: str,
    quantized_model: torch.nn.Module,
    output_path: Optional[str] = None,
) -> str:
    """Save quantized model to checkpoint."""
    # Load original checkpoint to preserve metadata
    original = torch.load(original_checkpoint_path, map_location="cpu")

    if output_path is None:
        # Add _int8 suffix to filename
        base = original_checkpoint_path.rsplit(".", 1)[0]
        output_path = f"{base}_int8.pth"

    # Create new checkpoint with quantized model
    checkpoint = {
        "model_state_dict": quantized_model.state_dict(),
        "board_type": original.get("board_type", "square8"),
        "hidden_dim": original.get("hidden_dim", 256),
        "num_hidden_layers": original.get("num_hidden_layers", 2),
        "architecture_version": original.get("architecture_version", "v1.0.0"),
        "quantized": True,
        "quantization_dtype": "int8",
        "original_checkpoint": original_checkpoint_path,
    }

    # Preserve training metadata if present
    for key in ["epoch", "val_loss", "best_epoch", "created_at"]:
        if key in original:
            checkpoint[key] = original[key]

    torch.save(checkpoint, output_path)
    return output_path


def benchmark_model(model_path: str, num_iterations: int = 1000) -> dict:
    """Benchmark model inference speed."""
    model = load_model(model_path)
    board_type = model.board_type
    feature_dim = get_feature_dim(board_type)

    # Create random input
    features = np.random.randn(feature_dim).astype(np.float32)

    # Benchmark original model
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(features[None, ...]).float()

        # Warmup
        for _ in range(100):
            _ = model(x)

        # Timed run
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = model(x)
        original_time = time.perf_counter() - start

    # Quantize and benchmark
    quantized = quantize_model(model)
    with torch.no_grad():
        # Warmup
        for _ in range(100):
            _ = quantized(x)

        # Timed run
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = quantized(x)
        quantized_time = time.perf_counter() - start

    # Measure accuracy difference
    with torch.no_grad():
        original_output = model(x).item()
        quantized_output = quantized(x).item()
        accuracy_diff = abs(original_output - quantized_output)

    return {
        "original_time_ms": (original_time / num_iterations) * 1000,
        "quantized_time_ms": (quantized_time / num_iterations) * 1000,
        "speedup": original_time / quantized_time,
        "original_output": original_output,
        "quantized_output": quantized_output,
        "output_diff": accuracy_diff,
    }


def find_nnue_models(directory: str) -> List[str]:
    """Find all NNUE model checkpoints in a directory."""
    models = []
    for ext in ["*.pth", "*.pt"]:
        for path in Path(directory).glob(f"**/{ext}"):
            # Skip already quantized models
            if "_int8" in str(path):
                continue
            # Check if it's an NNUE model
            if "nnue" in str(path).lower():
                models.append(str(path))
    return sorted(models)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize NNUE models to int8 for faster inference"
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        help="Path to model checkpoint or directory (with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Quantize all NNUE models in directory",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for quantized model",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark quantization speedup",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations for benchmark (default: 1000)",
    )

    args = parser.parse_args()

    if not args.model_path:
        parser.print_help()
        return 1

    if args.benchmark:
        print(f"Benchmarking {args.model_path}...")
        results = benchmark_model(args.model_path, args.iterations)
        print(f"\nResults ({args.iterations} iterations):")
        print(f"  Original:  {results['original_time_ms']:.3f} ms/inference")
        print(f"  Quantized: {results['quantized_time_ms']:.3f} ms/inference")
        print(f"  Speedup:   {results['speedup']:.2f}x")
        print(f"\nAccuracy:")
        print(f"  Original output:  {results['original_output']:.6f}")
        print(f"  Quantized output: {results['quantized_output']:.6f}")
        print(f"  Difference:       {results['output_diff']:.6f}")
        return 0

    if args.all:
        models = find_nnue_models(args.model_path)
        if not models:
            print(f"No NNUE models found in {args.model_path}")
            return 1

        print(f"Found {len(models)} NNUE models to quantize")
        for model_path in models:
            print(f"\nQuantizing {model_path}...")
            try:
                model = load_model(model_path)
                quantized = quantize_model(model)
                output_path = save_quantized_model(model_path, quantized)
                print(f"  Saved to {output_path}")
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print(f"Quantizing {args.model_path}...")
        model = load_model(args.model_path)
        quantized = quantize_model(model)
        output_path = save_quantized_model(args.model_path, quantized, args.output)
        print(f"Saved quantized model to {output_path}")

        # Show size comparison
        original_size = os.path.getsize(args.model_path) / 1024
        quantized_size = os.path.getsize(output_path) / 1024
        print(f"\nSize comparison:")
        print(f"  Original:  {original_size:.1f} KB")
        print(f"  Quantized: {quantized_size:.1f} KB")
        print(f"  Reduction: {(1 - quantized_size/original_size)*100:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
