# Hex Board Data Augmentation (D6 Symmetry)

> **Last Updated**: 2025-12-17
> **Status**: Active
> **Location**: `app/training/hex_augmentation.py`

This document describes the D6 dihedral symmetry augmentation system for hexagonal board games.

## Table of Contents

1. [Overview](#overview)
2. [D6 Dihedral Group](#d6-dihedral-group)
3. [Supported Board Sizes](#supported-board-sizes)
4. [Usage](#usage)
5. [API Reference](#api-reference)
6. [Implementation Details](#implementation-details)

---

## Overview

Hexagonal boards have 6-fold rotational symmetry (D6 dihedral group), providing 12 equivalent transformations. Data augmentation exploits this symmetry to effectively multiply dataset size by 12x without collecting additional games.

### Benefits

- **12x effective dataset size**: Each game provides 12 training samples
- **Better generalization**: Model learns position equivalence
- **Reduced overfitting**: More diverse training data
- **No additional compute for data collection**: Augmentation is applied on-the-fly

### Applicability

| Board Type        | Augmentation       | Symmetry Group              |
| ----------------- | ------------------ | --------------------------- |
| hex8 (9x9)        | D6 (12 transforms) | 6 rotations + 6 reflections |
| hexagonal (25x25) | D6 (12 transforms) | 6 rotations + 6 reflections |
| square8, square19 | D4 (8 transforms)  | Different implementation    |

---

## D6 Dihedral Group

The D6 group consists of 12 elements:

### Rotations (R)

| ID  | Rotation | Angle | Description |
| --- | -------- | ----- | ----------- |
| 0   | R^0      | 0°    | Identity    |
| 1   | R^1      | 60°   | Rotate CCW  |
| 2   | R^2      | 120°  | Rotate CCW  |
| 3   | R^3      | 180°  | Rotate CCW  |
| 4   | R^4      | 240°  | Rotate CCW  |
| 5   | R^5      | 300°  | Rotate CCW  |

### Reflections (S)

| ID  | Reflection | Description                 |
| --- | ---------- | --------------------------- |
| 6   | S          | Reflect across primary axis |
| 7   | S \* R^1   | Reflect + rotate 60°        |
| 8   | S \* R^2   | Reflect + rotate 120°       |
| 9   | S \* R^3   | Reflect + rotate 180°       |
| 10  | S \* R^4   | Reflect + rotate 240°       |
| 11  | S \* R^5   | Reflect + rotate 300°       |

### Group Properties

- **Closure**: Composing any two transforms yields another transform
- **Identity**: Transform 0 is the identity
- **Inverse**: Every transform has an inverse
  - Rotation inverses: R^k inverse is R^(6-k)
  - Reflections are self-inverse

---

## Supported Board Sizes

### Hex8 (Radius-4)

```
Board Size: 9x9 bounding box
Hex Radius: 4
Total Cells: 61
Policy Size: ~4,500 actions
Max Distance: 8 (diameter)
```

### Hexagonal (Radius-12)

```
Board Size: 25x25 bounding box
Hex Radius: 12
Total Cells: 469
Policy Size: ~92,000 actions
Max Distance: 24 (diameter)
```

### Policy Layout

The policy vector is organized as:

```
[0, placement_span)           : Placement actions
[movement_base, special_base) : Movement actions
[special_base]                : Skip/pass action
```

Each transformation must correctly map indices within each section.

---

## Usage

### Training Configuration

Enable hex augmentation in the unified loop config:

```yaml
# config/unified_loop.yaml
training:
  use_hex_augmentation: true
```

### CLI Usage

```bash
python -m app.training.train \
  --data-path data/training/hex8_games.npz \
  --board-type hex8 \
  --augment-hex-symmetry
```

### Programmatic Usage

```python
from app.training.hex_augmentation import (
    HexSymmetryTransform,
    augment_hex_sample,
)

# Create transform for hex8 board
transform = HexSymmetryTransform(board_size=9)

# Apply specific transformation
transformed_board = transform.transform_board(features, transform_id=3)
transformed_policy = transform.transform_policy(policy, transform_id=3)

# Apply all 12 transformations to a sample
augmented_samples = augment_hex_sample(
    features=features,
    globals_vec=globals_vec,
    policy_indices=policy_indices,
    policy_values=policy_values,
    board_size=9,  # Optional, inferred from features
)
# Returns list of 12 (features, globals, policy_indices, policy_values) tuples
```

---

## API Reference

### HexSymmetryTransform Class

```python
class HexSymmetryTransform:
    """D6 dihedral symmetry transformations for hexagonal boards."""

    def __init__(self, board_size: int = 25):
        """
        Initialize transform for given board size.

        Args:
            board_size: Bounding box size (9 for hex8, 25 for hexagonal)
        """

    @staticmethod
    def get_all_transforms() -> List[int]:
        """Returns list of all 12 transformation IDs [0, 1, ..., 11]."""

    def transform_board(self, board: np.ndarray, transform_id: int) -> np.ndarray:
        """
        Transform board features (C, H, W) tensor.

        Args:
            board: Feature tensor of shape (channels, height, width)
            transform_id: Transformation index 0-11

        Returns:
            Transformed feature tensor of same shape
        """

    def transform_policy(self, policy: np.ndarray, transform_id: int) -> np.ndarray:
        """
        Transform dense policy vector.

        Args:
            policy: Policy vector of length policy_size
            transform_id: Transformation index 0-11

        Returns:
            Transformed policy vector of same length
        """

    def transform_sparse_policy(
        self,
        policy_indices: np.ndarray,
        policy_values: np.ndarray,
        transform_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform sparse policy representation.

        Args:
            policy_indices: Array of policy indices
            policy_values: Array of policy values
            transform_id: Transformation index 0-11

        Returns:
            Tuple of (transformed_indices, transformed_values)
        """

    def transform_move_index(
        self,
        move_idx: int,
        transform_id: int,
    ) -> int:
        """
        Transform a single move index.

        Returns:
            Transformed move index, or -1 if out of bounds
        """

    def get_inverse_transform(self, transform_id: int) -> int:
        """
        Get the inverse of a transformation.

        For rotations R^k: inverse is R^(6-k)
        For reflections: self-inverse
        """

    @staticmethod
    def compose_transforms(t1: int, t2: int) -> int:
        """
        Compose two transformations: result = t2(t1(x))

        Used to verify group properties and find inverses.
        """
```

### Helper Functions

```python
def get_hex_policy_layout(board_size: int) -> dict:
    """
    Compute policy layout constants for a given hex board size.

    Args:
        board_size: Bounding box size (e.g., 25 for hex, 9 for hex8)

    Returns:
        Dictionary with:
        - placement_span: Size of placement action space
        - movement_base: Start index of movement actions
        - movement_span: Size of movement action space
        - special_base: Start index of special actions
        - policy_size: Total policy vector size
        - max_dist: Maximum movement distance
        - radius: Board radius
    """

def augment_hex_sample(
    features: np.ndarray,
    globals_vec: np.ndarray,
    policy_indices: np.ndarray,
    policy_values: np.ndarray,
    transform: Optional[HexSymmetryTransform] = None,
    board_size: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Apply all 12 D6 transformations to a single training sample.

    Returns:
        List of 12 tuples, each containing
        (features, globals, policy_indices, policy_values)
    """
```

---

## Implementation Details

### Coordinate Systems

The implementation uses two coordinate systems:

1. **Canonical (cx, cy)**: Standard 2D array indices
2. **Axial (q, r)**: Hex-native coordinates for transformations

Conversion functions:

```python
def _canonical_to_axial(cx, cy):
    q = cx - radius
    r = cy - radius
    return q, r

def _axial_to_canonical(q, r):
    cx = q + radius
    cy = r + radius
    return cx, cy
```

### Rotation in Axial Coordinates

60° CCW rotation:

```python
def rotate_60_ccw(q, r):
    return -r, q + r
```

### Reflection

Primary reflection (across q-axis):

```python
def reflect(q, r):
    return q, -r - q  # or equivalent: return -r - q, r
```

### Direction Mapping

Movement directions must also be transformed. The 6 hex directions:

```python
HEX_DIRS = [
    (1, 0),    # +q (East)
    (0, 1),    # +r (Southeast)
    (-1, 1),   # -q+r (Southwest)
    (-1, 0),   # -q (West)
    (0, -1),   # -r (Northwest)
    (1, -1),   # +q-r (Northeast)
]
```

Each rotation shifts direction indices by 1, reflections reverse the ordering.

### Performance Optimization

The implementation uses precomputed index maps for efficiency:

```python
# Precompute once
self._coord_maps[transform_id] = (cx_map, cy_map)  # For board transforms
self._dir_maps[transform_id] = dir_permutation     # For direction transforms

# Apply using numpy advanced indexing
transformed = board[:, cy_map, cx_map]  # Very fast
```

---

## Training Dataset Integration

### On-the-fly Augmentation

The `RingRiftDataset` class applies augmentation during training:

```python
# In app/training/train.py
class RingRiftDataset(Dataset):
    def __init__(self, ..., augment_hex: bool = False):
        if self.augment_hex:
            hex_board_size = 9 if board_type == BoardType.HEX8 else 25
            self.hex_transform = HexSymmetryTransform(board_size=hex_board_size)

    def __getitem__(self, idx):
        # ... load sample ...

        if self.augment_hex and self.hex_transform is not None:
            # Random transform from D6 group
            transform_id = random.randint(0, 11)
            if transform_id != 0:
                features = self.hex_transform.transform_board(features, transform_id)
                policy_indices, policy_values = self.hex_transform.transform_sparse_policy(
                    policy_indices, policy_values, transform_id
                )

        return features, globals_vec, policy, value
```

### Epoch Multiplier

With on-the-fly augmentation, each epoch sees different transformations. Over multiple epochs, all 12 transformations are sampled with equal probability.

---

## Verification

The implementation includes verification of group properties:

```python
# Verify composition closure
for t1 in range(12):
    for t2 in range(12):
        t_composed = compose_transforms(t1, t2)
        assert 0 <= t_composed < 12  # Closure

# Verify inverses
for t in range(12):
    t_inv = get_inverse_transform(t)
    assert compose_transforms(t, t_inv) == 0  # Identity
```

---

## See Also

- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Training configuration
- [NEURAL_AI_ARCHITECTURE.md](NEURAL_AI_ARCHITECTURE.md) - HexNeuralNet architectures
- [GAME_NOTATION_SPEC.md](GAME_NOTATION_SPEC.md) - Move encoding specification
