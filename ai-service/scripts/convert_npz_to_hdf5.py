#!/usr/bin/env python3
"""Convert NPZ training data files to HDF5 format for better performance.

HDF5 provides faster batch loading via native fancy indexing support,
which can improve training throughput by 15-25% compared to NPZ files.

Usage:
    # Convert a single file
    python scripts/convert_npz_to_hdf5.py input.npz output.h5

    # Convert all NPZ files in a directory
    python scripts/convert_npz_to_hdf5.py --input-dir data/games --output-dir data/games_hdf5

    # Convert with compression (slower to create but smaller files)
    python scripts/convert_npz_to_hdf5.py input.npz output.h5 --compress

    # Verify conversion integrity
    python scripts/convert_npz_to_hdf5.py input.npz output.h5 --verify
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.lib.logging_config import setup_script_logging

# Try to import h5py
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

logger = setup_script_logging("convert_npz_to_hdf5")


def compute_array_checksum(arr: np.ndarray) -> str:
    """Compute SHA256 checksum of numpy array."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def convert_npz_to_hdf5(
    npz_path: Path,
    hdf5_path: Path,
    compress: bool = False,
    chunk_size: int = 1000,
    verify: bool = False,
) -> dict[str, Any]:
    """Convert an NPZ file to HDF5 format.

    Args:
        npz_path: Path to input NPZ file
        hdf5_path: Path to output HDF5 file
        compress: Enable gzip compression (smaller files, slower read)
        chunk_size: HDF5 chunk size for datasets
        verify: Verify conversion by comparing checksums

    Returns:
        Dict with conversion statistics
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for HDF5 conversion. Install with: pip install h5py")

    if not npz_path.exists():
        raise FileNotFoundError(f"Input file not found: {npz_path}")

    start_time = time.time()
    logger.info(f"Converting {npz_path} -> {hdf5_path}")

    # Load NPZ file
    npz_data = np.load(npz_path, allow_pickle=True)
    keys = list(npz_data.keys())

    # Get sample count from features array
    if 'features' not in npz_data:
        raise ValueError(f"NPZ file missing 'features' key: {npz_path}")

    num_samples = len(npz_data['features'])
    logger.info(f"  Samples: {num_samples}, Keys: {keys}")

    # Create output directory if needed
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    # Compression settings
    compression = 'gzip' if compress else None
    compression_opts = 4 if compress else None

    checksums_npz: dict[str, str] = {}
    checksums_hdf5: dict[str, str] = {}

    with h5py.File(hdf5_path, 'w') as hf:
        # Store metadata
        hf.attrs['source_file'] = str(npz_path)
        hf.attrs['num_samples'] = num_samples
        hf.attrs['converted_at'] = time.time()
        hf.attrs['format_version'] = '1.0'

        for key in keys:
            arr = npz_data[key]

            # Handle string arrays (like board_type, policy_encoding)
            if arr.dtype.kind in ('U', 'S'):  # Unicode or byte string
                # Convert to variable-length UTF-8 strings
                dt = h5py.special_dtype(vlen=str)

                # Scalar strings (0-dim arrays) need special handling
                if arr.ndim == 0:
                    ds = hf.create_dataset(key, data=str(arr.item()), dtype=dt)
                    logger.info(f"  {key}: scalar string")
                else:
                    ds = hf.create_dataset(key, shape=arr.shape, dtype=dt)
                    for i in range(len(arr)):
                        ds[i] = str(arr[i])
                    logger.info(f"  {key}: string array {arr.shape}")

                if verify:
                    # For string arrays, compute checksum of joined strings
                    if arr.ndim == 0:
                        joined = str(arr.item())
                    else:
                        joined = ''.join(str(x) for x in arr.flatten())
                    checksums_npz[key] = compute_array_checksum(
                        np.frombuffer(joined.encode('utf-8'), dtype=np.uint8)
                    )

            # Handle object arrays (sparse policies or string arrays)
            elif arr.dtype == object:
                # Flatten 2D object arrays with shape (n, 1) to 1D
                arr_flat = arr.flatten() if arr.ndim > 1 else arr

                # Check if this is a string object array (like phases, victory_types)
                first_item = None
                for item in arr_flat:
                    if item is not None:
                        first_item = item
                        break

                is_string_array = isinstance(first_item, str)

                if is_string_array:
                    # Store as variable-length string dataset
                    dt = h5py.special_dtype(vlen=str)
                    ds = hf.create_dataset(key, shape=(len(arr_flat),), dtype=dt)

                    for i, item in enumerate(arr_flat):
                        ds[i] = str(item) if item is not None else ""

                    logger.info(f"  {key}: string object array ({len(arr_flat)} items)")

                    if verify:
                        joined = ''.join(str(x) for x in arr_flat if x is not None)
                        checksums_npz[key] = compute_array_checksum(
                            np.frombuffer(joined.encode('utf-8'), dtype=np.uint8)
                        )
                else:
                    # Numeric variable-length arrays (sparse policies)
                    # Determine appropriate dtype
                    if 'indices' in key:
                        dt = h5py.special_dtype(vlen=np.int32)
                    elif 'values' in key:
                        dt = h5py.special_dtype(vlen=np.float32)
                    else:
                        # Try to infer from first non-empty element
                        dt = h5py.special_dtype(vlen=np.float32)
                        for item in arr_flat:
                            arr_item = np.asarray(item)
                            if arr_item.size > 0:
                                dt = h5py.special_dtype(vlen=arr_item.dtype)
                                break

                    ds = hf.create_dataset(
                        key,
                        shape=(len(arr_flat),),
                        dtype=dt,
                    )

                    # Copy data - use np.asarray().size to handle both scalars and arrays
                    for i, item in enumerate(arr_flat):
                        arr_item = np.asarray(item)
                        if arr_item.size > 0:
                            # Ensure it's at least 1D for vlen storage
                            ds[i] = arr_item.flatten()
                        else:
                            # Empty array - store as empty
                            ds[i] = np.array([], dtype=ds.dtype.metadata['vlen'])

                    logger.info(f"  {key}: variable-length array ({len(arr_flat)} items)")

                    if verify:
                        # For object arrays, compute checksum of concatenated data
                        non_empty = [np.asarray(x).flatten() for x in arr_flat
                                     if np.asarray(x).size > 0]
                        if non_empty:
                            concat = np.concatenate(non_empty)
                            checksums_npz[key] = compute_array_checksum(concat.astype(np.float32))
                        else:
                            checksums_npz[key] = compute_array_checksum(np.array([], dtype=np.float32))

            # Handle scalar numeric values (ndim == 0)
            elif arr.ndim == 0:
                # Scalar datasets cannot use chunking or compression
                hf.create_dataset(key, data=arr[()])
                logger.info(f"  {key}: scalar {arr.dtype}")

                if verify:
                    checksums_npz[key] = compute_array_checksum(np.asarray([arr[()]]))

            else:
                # Regular dense array
                arr_np = np.asarray(arr)

                # Determine chunk shape
                if len(arr_np.shape) == 1:
                    chunks = (min(chunk_size, num_samples),)
                else:
                    chunks = (min(chunk_size, num_samples),) + arr_np.shape[1:]

                ds = hf.create_dataset(
                    key,
                    data=arr_np,
                    chunks=chunks,
                    compression=compression,
                    compression_opts=compression_opts,
                )

                size_mb = arr_np.nbytes / (1024 * 1024)
                logger.info(f"  {key}: {arr_np.shape} {arr_np.dtype} ({size_mb:.1f} MB)")

                if verify:
                    checksums_npz[key] = compute_array_checksum(arr_np)

    npz_data.close()

    # Get file sizes
    npz_size = npz_path.stat().st_size
    hdf5_size = hdf5_path.stat().st_size
    duration = time.time() - start_time

    result = {
        'success': True,
        'npz_path': str(npz_path),
        'hdf5_path': str(hdf5_path),
        'num_samples': num_samples,
        'num_keys': len(keys),
        'npz_size_mb': npz_size / (1024 * 1024),
        'hdf5_size_mb': hdf5_size / (1024 * 1024),
        'compression_ratio': npz_size / hdf5_size if hdf5_size > 0 else 0,
        'duration_seconds': duration,
    }

    # Verify if requested
    if verify:
        logger.info("Verifying conversion...")
        with h5py.File(hdf5_path, 'r') as hf:
            for key in keys:
                arr = hf[key]
                # Check scalar first (before vlen check, as scalar strings have vlen metadata)
                if arr.shape == ():
                    # Scalar value (numeric or string)
                    val = arr[()]
                    if isinstance(val, (str, bytes)):
                        s = val if isinstance(val, str) else val.decode('utf-8')
                        checksums_hdf5[key] = compute_array_checksum(
                            np.frombuffer(s.encode('utf-8'), dtype=np.uint8)
                        )
                    else:
                        checksums_hdf5[key] = compute_array_checksum(np.asarray([val]))
                elif arr.dtype.metadata and 'vlen' in arr.dtype.metadata:
                    vlen_type = arr.dtype.metadata['vlen']
                    if vlen_type == str:
                        # Variable-length string array - decode bytes if needed
                        strings = []
                        for i in range(len(arr)):
                            val = arr[i]
                            if val:
                                if isinstance(val, bytes):
                                    val = val.decode('utf-8')
                                strings.append(val)
                        joined = ''.join(strings)
                        checksums_hdf5[key] = compute_array_checksum(
                            np.frombuffer(joined.encode('utf-8'), dtype=np.uint8)
                        )
                    else:
                        # Variable-length numeric array
                        non_empty = [np.asarray(arr[i]).flatten() for i in range(len(arr))
                                     if arr[i] is not None and len(arr[i]) > 0]
                        if non_empty:
                            concat = np.concatenate(non_empty)
                            checksums_hdf5[key] = compute_array_checksum(concat.astype(np.float32))
                        else:
                            checksums_hdf5[key] = compute_array_checksum(np.array([], dtype=np.float32))
                else:
                    checksums_hdf5[key] = compute_array_checksum(np.asarray(arr[:]))

        # Compare checksums
        mismatches = []
        for key in keys:
            if key in checksums_npz and key in checksums_hdf5:
                if checksums_npz[key] != checksums_hdf5[key]:
                    mismatches.append(key)

        if mismatches:
            result['success'] = False
            result['verification_failed'] = mismatches
            logger.error(f"Verification FAILED for keys: {mismatches}")
        else:
            result['verified'] = True
            logger.info("Verification PASSED - all checksums match")

    logger.info(
        f"Conversion complete: {result['npz_size_mb']:.1f}MB -> {result['hdf5_size_mb']:.1f}MB "
        f"({result['compression_ratio']:.2f}x) in {duration:.1f}s"
    )

    return result


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    compress: bool = False,
    verify: bool = False,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Convert all NPZ files in a directory to HDF5.

    Args:
        input_dir: Directory containing NPZ files
        output_dir: Output directory for HDF5 files
        compress: Enable compression
        verify: Verify each conversion
        skip_existing: Skip files that already have HDF5 versions

    Returns:
        Dict with overall statistics
    """
    npz_files = sorted(input_dir.glob("*.npz"))

    if not npz_files:
        logger.warning(f"No NPZ files found in {input_dir}")
        return {'success': False, 'error': 'No NPZ files found'}

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = 0
    failed = 0

    for npz_path in npz_files:
        hdf5_path = output_dir / (npz_path.stem + ".h5")

        if skip_existing and hdf5_path.exists():
            logger.info(f"Skipping {npz_path.name} (HDF5 exists)")
            skipped += 1
            continue

        try:
            result = convert_npz_to_hdf5(
                npz_path=npz_path,
                hdf5_path=hdf5_path,
                compress=compress,
                verify=verify,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to convert {npz_path}: {e}")
            failed += 1

    total_npz_mb = sum(r['npz_size_mb'] for r in results)
    total_hdf5_mb = sum(r['hdf5_size_mb'] for r in results)
    total_samples = sum(r['num_samples'] for r in results)

    return {
        'success': failed == 0,
        'files_converted': len(results),
        'files_skipped': skipped,
        'files_failed': failed,
        'total_samples': total_samples,
        'total_npz_mb': total_npz_mb,
        'total_hdf5_mb': total_hdf5_mb,
        'overall_compression_ratio': total_npz_mb / total_hdf5_mb if total_hdf5_mb > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert NPZ training data to HDF5 format for faster loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        'input',
        nargs='?',
        help='Input NPZ file (or use --input-dir for directory mode)',
    )
    parser.add_argument(
        'output',
        nargs='?',
        help='Output HDF5 file (or use --output-dir for directory mode)',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        help='Input directory containing NPZ files',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for HDF5 files',
    )
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Enable gzip compression (smaller files, slower read)',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify conversion integrity via checksums',
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Overwrite existing HDF5 files in directory mode',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='HDF5 chunk size (default: 1000)',
    )

    args = parser.parse_args()

    if not HAS_H5PY:
        print("ERROR: h5py is required. Install with: pip install h5py")
        sys.exit(1)

    # Directory mode
    if args.input_dir:
        if not args.output_dir:
            args.output_dir = args.input_dir.parent / (args.input_dir.name + "_hdf5")

        result = convert_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            compress=args.compress,
            verify=args.verify,
            skip_existing=not args.no_skip_existing,
        )

        print(f"\nConversion Summary:")
        print(f"  Files converted: {result['files_converted']}")
        print(f"  Files skipped: {result['files_skipped']}")
        print(f"  Files failed: {result['files_failed']}")
        print(f"  Total samples: {result['total_samples']:,}")
        print(f"  Total size: {result['total_npz_mb']:.1f}MB -> {result['total_hdf5_mb']:.1f}MB")
        print(f"  Compression ratio: {result['overall_compression_ratio']:.2f}x")

        sys.exit(0 if result['success'] else 1)

    # Single file mode
    if not args.input or not args.output:
        parser.print_help()
        print("\nError: Provide input and output files, or use --input-dir for directory mode")
        sys.exit(1)

    result = convert_npz_to_hdf5(
        npz_path=Path(args.input),
        hdf5_path=Path(args.output),
        compress=args.compress,
        chunk_size=args.chunk_size,
        verify=args.verify,
    )

    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
