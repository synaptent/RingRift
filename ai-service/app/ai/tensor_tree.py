"""Tensor-based MCTS tree storage for GPU-accelerated search.

This module provides a TensorTree data structure that stores MCTS tree nodes
as GPU tensors, enabling vectorized operations for selection, expansion,
and backpropagation.

Key optimizations (Phase 2 of GPU MCTS plan):
1. Structure-of-Arrays pattern for cache-efficient access
2. Vectorized UCB computation across all children
3. Scatter-based backpropagation for parallel value updates
4. Pre-allocated tensors to avoid dynamic memory allocation

Based on research from:
- TurboZero: Vectorized PyTorch MCTS
- AlphaZero.jl: Julia GPU MCTS implementation (13x speedup)
- MCTS-NC: Numba CUDA parallelization

Usage:
    tree = TensorTree(
        max_nodes=10000,
        max_children=100,  # Max children per node
        device="cuda",
    )

    # Add root node
    root_idx = tree.add_root(prior=policy_logits)

    # Vectorized selection
    leaf_indices = tree.select_batch(root_indices, c_puct=1.5)

    # Vectorized backpropagation
    tree.backpropagate_batch(leaf_indices, values)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TensorTree:
    """MCTS tree stored as tensors for vectorized GPU operations.

    The tree structure uses a flat array representation where:
    - Each node has a unique index (0 to num_nodes-1)
    - Parent relationships are stored as indices
    - Children are stored as contiguous blocks with start/count

    This enables vectorized operations across all nodes/paths simultaneously.

    Attributes:
        max_nodes: Maximum number of nodes in tree
        max_children: Maximum children per node
        device: GPU device for tensors
    """

    # Tree structure tensors
    parent_idx: torch.Tensor      # (max_nodes,) Parent node index, -1 for root
    children_start: torch.Tensor  # (max_nodes,) Index of first child in children array
    children_count: torch.Tensor  # (max_nodes,) Number of children
    depth: torch.Tensor           # (max_nodes,) Depth from root (root=0)

    # Node statistics tensors
    visit_count: torch.Tensor     # (max_nodes,) N(s,a)
    total_value: torch.Tensor     # (max_nodes,) W(s,a) sum of values
    prior: torch.Tensor           # (max_nodes,) P(s,a) from policy network

    # Children array (flattened)
    # children[children_start[i]:children_start[i]+children_count[i]] = indices of node i's children
    children: torch.Tensor        # (max_nodes * max_children,) Child node indices

    # Move encoding for each node (to reconstruct path)
    move_idx: torch.Tensor        # (max_nodes,) Move index that led to this node

    # Metadata
    num_nodes: int                # Current number of nodes in tree
    max_nodes: int
    max_children: int
    device: torch.device

    @classmethod
    def create(
        cls,
        max_nodes: int = 10000,
        max_children: int = 100,
        device: torch.device | str = "cuda",
    ) -> TensorTree:
        """Create an empty TensorTree with pre-allocated tensors.

        Args:
            max_nodes: Maximum number of nodes (default 10000)
            max_children: Maximum children per node (default 100)
            device: GPU device (default "cuda")

        Returns:
            Empty TensorTree ready for use
        """
        if isinstance(device, str):
            device = torch.device(device)

        return cls(
            # Tree structure
            parent_idx=torch.full((max_nodes,), -1, dtype=torch.int32, device=device),
            children_start=torch.zeros(max_nodes, dtype=torch.int32, device=device),
            children_count=torch.zeros(max_nodes, dtype=torch.int32, device=device),
            depth=torch.zeros(max_nodes, dtype=torch.int16, device=device),

            # Node statistics
            visit_count=torch.zeros(max_nodes, dtype=torch.int32, device=device),
            total_value=torch.zeros(max_nodes, dtype=torch.float32, device=device),
            prior=torch.zeros(max_nodes, dtype=torch.float32, device=device),

            # Children array
            children=torch.full(
                (max_nodes * max_children,), -1,
                dtype=torch.int32, device=device
            ),

            # Move encoding
            move_idx=torch.full((max_nodes,), -1, dtype=torch.int32, device=device),

            # Metadata
            num_nodes=0,
            max_nodes=max_nodes,
            max_children=max_children,
            device=device,
        )

    def reset(self) -> None:
        """Reset tree to empty state (reuse allocated memory)."""
        self.parent_idx.fill_(-1)
        self.children_start.zero_()
        self.children_count.zero_()
        self.depth.zero_()
        self.visit_count.zero_()
        self.total_value.zero_()
        self.prior.zero_()
        self.children.fill_(-1)
        self.move_idx.fill_(-1)
        self.num_nodes = 0

    def add_root(self, prior: torch.Tensor | None = None) -> int:
        """Add root node to the tree.

        Args:
            prior: Prior probability for root (usually uniform)

        Returns:
            Index of root node (always 0)
        """
        if self.num_nodes > 0:
            raise ValueError("Tree already has a root. Call reset() first.")

        self.parent_idx[0] = -1  # No parent
        self.depth[0] = 0
        if prior is not None:
            self.prior[0] = prior if prior.dim() == 0 else prior.mean()

        self.num_nodes = 1
        return 0

    def expand_node(
        self,
        node_idx: int,
        priors: torch.Tensor,
        move_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Expand a node by adding children.

        Args:
            node_idx: Index of node to expand
            priors: (num_children,) Prior probabilities for each child
            move_indices: (num_children,) Move indices for each child

        Returns:
            Tensor of new child node indices
        """
        num_children = len(priors)
        if num_children == 0:
            return torch.tensor([], dtype=torch.int32, device=self.device)

        if self.num_nodes + num_children > self.max_nodes:
            raise RuntimeError(
                f"Tree full: {self.num_nodes} + {num_children} > {self.max_nodes}"
            )

        # Allocate child node indices
        start_idx = self.num_nodes
        end_idx = start_idx + num_children
        child_indices = torch.arange(
            start_idx, end_idx,
            dtype=torch.int32, device=self.device
        )

        # Set parent relationship
        self.parent_idx[start_idx:end_idx] = node_idx
        self.depth[start_idx:end_idx] = self.depth[node_idx] + 1

        # Set priors and move indices
        self.prior[start_idx:end_idx] = priors.to(self.device)
        self.move_idx[start_idx:end_idx] = move_indices.to(self.device)

        # Update children array for parent
        children_offset = node_idx * self.max_children
        self.children_start[node_idx] = children_offset
        self.children_count[node_idx] = num_children
        self.children[children_offset:children_offset + num_children] = child_indices

        self.num_nodes = end_idx
        return child_indices

    def get_children(self, node_idx: int) -> torch.Tensor:
        """Get child indices for a node.

        Args:
            node_idx: Index of parent node

        Returns:
            Tensor of child node indices
        """
        start = self.children_start[node_idx].item()
        count = self.children_count[node_idx].item()
        if count == 0:
            return torch.tensor([], dtype=torch.int32, device=self.device)
        return self.children[start:start + count]

    def compute_ucb_batch(
        self,
        node_indices: torch.Tensor,
        c_puct: float = 1.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute UCB scores for all children of given nodes (vectorized).

        Uses the PUCT formula from AlphaZero:
        UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)

        Args:
            node_indices: (batch_size,) Indices of parent nodes
            c_puct: Exploration constant

        Returns:
            Tuple of (best_child_indices, ucb_scores) where:
            - best_child_indices: (batch_size,) Best child for each parent
            - ucb_scores: (batch_size, max_children) UCB scores
        """
        batch_size = node_indices.shape[0]
        device = self.device

        # Get parent visit counts
        parent_visits = self.visit_count[node_indices].float()  # (batch_size,)
        parent_sqrt = torch.sqrt(parent_visits + 1)  # (batch_size,)

        # Initialize UCB scores with -inf (for invalid children)
        ucb_scores = torch.full(
            (batch_size, self.max_children),
            float('-inf'),
            dtype=torch.float32,
            device=device,
        )

        # Get children info for each parent
        children_starts = self.children_start[node_indices]  # (batch_size,)
        children_counts = self.children_count[node_indices]  # (batch_size,)

        # For each potential child position
        for c in range(self.max_children):
            # Mask for parents that have this child
            has_child = c < children_counts

            if not has_child.any():
                break

            # Get child indices
            child_offsets = children_starts + c
            child_indices = self.children[child_offsets.long()]

            # Get child stats (masked to valid children only)
            child_visits = torch.where(
                has_child,
                self.visit_count[child_indices.long()].float(),
                torch.zeros(batch_size, device=device),
            )
            child_values = torch.where(
                has_child,
                self.total_value[child_indices.long()],
                torch.zeros(batch_size, device=device),
            )
            child_priors = torch.where(
                has_child,
                self.prior[child_indices.long()],
                torch.zeros(batch_size, device=device),
            )

            # Compute Q values (mean value)
            q_values = torch.where(
                child_visits > 0,
                child_values / child_visits,
                torch.zeros(batch_size, device=device),
            )

            # Compute UCB: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
            exploration = c_puct * child_priors * parent_sqrt / (1 + child_visits)
            ucb = q_values + exploration

            # Store UCB scores (only for valid children)
            ucb_scores[:, c] = torch.where(has_child, ucb, ucb_scores[:, c])

        # Find best child for each parent
        best_child_positions = ucb_scores.argmax(dim=1)  # (batch_size,)

        # Convert positions to actual child indices
        best_child_offsets = children_starts + best_child_positions
        best_child_indices = self.children[best_child_offsets.long()]

        return best_child_indices, ucb_scores

    def select_batch(
        self,
        root_indices: torch.Tensor,
        c_puct: float = 1.5,
        max_depth: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select leaf nodes from multiple roots using UCB (vectorized).

        Performs tree traversal from each root to a leaf, selecting children
        with highest UCB scores at each step.

        Args:
            root_indices: (batch_size,) Starting node indices
            c_puct: Exploration constant for UCB
            max_depth: Maximum traversal depth

        Returns:
            Tuple of (leaf_indices, paths) where:
            - leaf_indices: (batch_size,) Selected leaf node indices
            - paths: (batch_size, max_depth) Node indices along each path
        """
        batch_size = root_indices.shape[0]
        device = self.device

        # Track current positions and paths
        current = root_indices.clone()
        paths = torch.full(
            (batch_size, max_depth), -1,
            dtype=torch.int32, device=device
        )
        paths[:, 0] = root_indices

        for depth in range(1, max_depth):
            # Check which nodes have children (are not leaves)
            has_children = self.children_count[current.long()] > 0

            if not has_children.any():
                break

            # Select best children for non-leaf nodes
            best_children, _ = self.compute_ucb_batch(current, c_puct)

            # Update current positions (only for nodes with children)
            current = torch.where(has_children, best_children, current)
            paths[:, depth] = torch.where(
                has_children, current,
                torch.tensor(-1, device=device, dtype=torch.int32)
            )

        return current, paths

    def backpropagate_batch(
        self,
        leaf_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Backpropagate values from leaves to roots (vectorized).

        Uses scatter_add for parallel value accumulation along all paths.

        Args:
            leaf_indices: (batch_size,) Leaf node indices
            values: (batch_size,) Values to backpropagate
        """
        batch_size = leaf_indices.shape[0]
        device = self.device

        # Trace paths back to root
        current = leaf_indices.clone()
        visited = torch.zeros(self.num_nodes, dtype=torch.int32, device=device)
        value_sum = torch.zeros(self.num_nodes, dtype=torch.float32, device=device)

        max_depth = 100  # Safety limit
        for _ in range(max_depth):
            # Increment visit counts
            visited.scatter_add_(
                0, current.long(),
                torch.ones(batch_size, dtype=torch.int32, device=device)
            )

            # Add values
            value_sum.scatter_add_(0, current.long(), values)

            # Move to parents
            parents = self.parent_idx[current.long()]

            # Check if all reached root
            at_root = parents < 0
            if at_root.all():
                break

            # Update current (stay at root for those who reached it)
            current = torch.where(at_root, current, parents)

        # Apply accumulated updates
        self.visit_count[:self.num_nodes] += visited[:self.num_nodes]
        self.total_value[:self.num_nodes] += value_sum[:self.num_nodes]

    def get_visit_distribution(self, node_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get visit count distribution over children.

        Args:
            node_idx: Index of parent node

        Returns:
            Tuple of (move_indices, visit_probs) for children
        """
        children = self.get_children(node_idx)
        if len(children) == 0:
            return (
                torch.tensor([], dtype=torch.int32, device=self.device),
                torch.tensor([], dtype=torch.float32, device=self.device),
            )

        visits = self.visit_count[children.long()].float()
        total = visits.sum()
        probs = visits / total if total > 0 else torch.zeros_like(visits)
        move_indices = self.move_idx[children.long()]

        return move_indices, probs

    def get_best_child(self, node_idx: int) -> int:
        """Get child with highest visit count.

        Args:
            node_idx: Index of parent node

        Returns:
            Index of best child, or -1 if no children
        """
        children = self.get_children(node_idx)
        if len(children) == 0:
            return -1

        visits = self.visit_count[children.long()]
        best_pos = visits.argmax()
        return children[best_pos].item()

    def get_stats(self) -> dict:
        """Get tree statistics.

        Returns:
            Dictionary with tree statistics
        """
        if self.num_nodes == 0:
            return {
                "num_nodes": 0,
                "max_depth": 0,
                "total_visits": 0,
                "avg_branching": 0.0,
            }

        active_nodes = self.num_nodes
        max_depth = self.depth[:active_nodes].max().item()
        total_visits = self.visit_count[:active_nodes].sum().item()
        avg_branching = self.children_count[:active_nodes].float().mean().item()

        return {
            "num_nodes": active_nodes,
            "max_depth": max_depth,
            "total_visits": total_visits,
            "avg_branching": avg_branching,
        }


class MultiTensorTree:
    """Multiple MCTS trees for parallel game evaluation.

    Manages N independent trees, one per game, enabling vectorized
    operations across all games simultaneously.
    """

    def __init__(
        self,
        num_trees: int,
        max_nodes_per_tree: int = 1000,
        max_children: int = 100,
        device: torch.device | str = "cuda",
    ):
        """Initialize multiple trees.

        Args:
            num_trees: Number of parallel trees
            max_nodes_per_tree: Maximum nodes per tree
            max_children: Maximum children per node
            device: GPU device
        """
        self.num_trees = num_trees
        self.max_nodes_per_tree = max_nodes_per_tree
        self.max_children = max_children

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Create individual trees
        self.trees = [
            TensorTree.create(
                max_nodes=max_nodes_per_tree,
                max_children=max_children,
                device=device,
            )
            for _ in range(num_trees)
        ]

        logger.debug(
            f"MultiTensorTree: {num_trees} trees, "
            f"{max_nodes_per_tree} nodes each, device={device}"
        )

    def reset_all(self) -> None:
        """Reset all trees."""
        for tree in self.trees:
            tree.reset()

    def add_roots(self, priors: list[torch.Tensor | None] | None = None) -> list[int]:
        """Add root nodes to all trees.

        Args:
            priors: Optional list of prior tensors, one per tree

        Returns:
            List of root indices (all 0s)
        """
        if priors is None:
            priors = [None] * self.num_trees

        return [
            tree.add_root(prior)
            for tree, prior in zip(self.trees, priors, strict=False)
        ]

    def select_leaves(self, c_puct: float = 1.5) -> list[int]:
        """Select leaf nodes from all trees.

        Args:
            c_puct: Exploration constant

        Returns:
            List of leaf indices, one per tree
        """
        leaves = []
        for tree in self.trees:
            if tree.num_nodes == 0:
                leaves.append(-1)
                continue

            root_idx = torch.tensor([0], dtype=torch.int32, device=self.device)
            leaf, _ = tree.select_batch(root_idx, c_puct)
            leaves.append(leaf[0].item())

        return leaves

    def backpropagate(self, leaf_indices: list[int], values: list[float]) -> None:
        """Backpropagate values through all trees.

        Args:
            leaf_indices: Leaf index for each tree
            values: Value for each tree
        """
        for tree, leaf_idx, value in zip(
            self.trees, leaf_indices, values, strict=False
        ):
            if leaf_idx < 0:
                continue

            leaf_tensor = torch.tensor(
                [leaf_idx], dtype=torch.int32, device=self.device
            )
            value_tensor = torch.tensor(
                [value], dtype=torch.float32, device=self.device
            )
            tree.backpropagate_batch(leaf_tensor, value_tensor)

    def get_best_moves(self) -> list[int]:
        """Get best move index for each tree.

        Returns:
            List of best move indices, one per tree
        """
        moves = []
        for tree in self.trees:
            if tree.num_nodes == 0:
                moves.append(-1)
                continue

            best_child = tree.get_best_child(0)
            if best_child < 0:
                moves.append(-1)
            else:
                moves.append(tree.move_idx[best_child].item())

        return moves
