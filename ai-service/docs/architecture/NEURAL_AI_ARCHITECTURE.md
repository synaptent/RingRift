# RingRift Neural AI Architecture

## Design Philosophy

This architecture combines proven techniques from:

- **AlphaZero** (MCTS + neural network guidance)
- **AlphaStar** (factored action spaces for combinatorial complexity)
- **Stockfish NNUE** (hybrid classical + neural evaluation)
- **KataGo** (auxiliary training targets, ownership prediction)

The key insight: RingRift's extreme branching factor (potentially 100,000+ capture sequences per turn) requires **hierarchical action decomposition** and **aggressive pruning** guided by both learned and handcrafted heuristics.

---

## 1. State Representation

### 1.1 Board Tensor Encoding

For a 19×19 board with up to 4 players, encode as a 3D tensor `[C, H, W]`:

```
Channels (per player P, from current player's POV):
├── Stack Ownership (4 channels)
│   └── stack_top_owner[p] = 1.0 if player p controls stack
├── Stack Height (6 channels)
│   └── height_bucket[b] = 1.0 if stack height in bucket b
│   └── Buckets: [1], [2], [3], [4-5], [6-8], [9+]
├── Cap Height (6 channels)
│   └── Same bucketing as stack height
├── Buried Rings (4 channels)
│   └── has_buried_ring[p] = 1.0 if any ring of player p is below top
├── Markers (4 channels)
│   └── marker_owner[p] = 1.0 if marker belongs to player p
├── Collapsed Territory (4 channels)
│   └── territory_owner[p] = 1.0 if collapsed territory belongs to p
├── Valid Moves Mask (1 channel)
│   └── can_move_from[x,y] = 1.0 if current player can move from here
├── Capture Targets Mask (1 channel)
│   └── can_capture_at[x,y] = 1.0 if enemy stack can be captured
├── Line Threat Mask (4 channels)
│   └── line_threat[p] = proximity to completing a 5-line
└── Disconnection Risk Mask (4 channels)
    └── disconnect_risk[p] = 1.0 if region may become disconnected

Total: ~42 channels for 4-player
```

### 1.2 Global Features Vector

```python
global_features = [
    # Per-player (4 × 5 = 20 features)
    rings_in_hand[p] / MAX_RINGS,
    rings_on_board[p] / MAX_RINGS,
    eliminated_rings[p] / MAX_RINGS,
    territory_spaces[p] / BOARD_SIZE,
    num_stacks_controlled[p] / MAX_STACKS,

    # Game state (5 features)
    move_number / MAX_MOVES,
    current_player_one_hot[4],
    phase_one_hot[3],  # placement, movement, capture

    # Board type (3 features)
    board_type_one_hot[3],  # square8, square19, hex
]
# Total: ~28 global features
```

### 1.3 Hex Board Handling

Use axial coordinates with padding. The GNN approach handles this naturally; for CNN, use hexagonal convolutions or treat as sparse grid with masking.

---

## 2. Action Space Decomposition

### 2.1 Hierarchical Action Structure

A complete RingRift turn decomposes into:

```
Turn
├── Phase 1: Placement (optional)
│   ├── placement_type: {none, new_stack, takeover}
│   ├── location: (x, y)
│   └── ring_count: {1, 2, 3}  # for new_stack only
│
├── Phase 2: Movement
│   ├── source: (x, y)
│   ├── direction: {N, NE, E, SE, S, SW, W, NW}  # 8 for square
│   └── distance: {1, 2, ..., max_legal}
│
├── Phase 3: Capture Chain (recursive)
│   ├── continue_capture: {yes, no}
│   ├── capture_target: (x, y)
│   └── capture_portion: {all, top_n}  # if splitting stack
│
└── Phase 4: Post-Processing
    ├── line_to_process: index or none
    ├── line_option: {option1, option2}
    ├── region_to_collapse: index or none
    └── self_elimination_choice: ring_index
```

### 2.2 Policy Head Architecture

Instead of outputting a single distribution over all possible turns, use **autoregressive factorization**:

```
P(turn) = P(placement) × P(movement | placement) × P(captures | ...) × P(postproc | ...)
```

Each sub-policy is a separate head:

```python
class PolicyHeads(nn.Module):
    def __init__(self, embed_dim, board_size, num_players):
        # Placement heads
        self.placement_type = nn.Linear(embed_dim, 3)  # none/new/takeover
        self.placement_location = nn.Conv2d(embed_dim, 1, 1)  # spatial
        self.placement_rings = nn.Linear(embed_dim, 3)  # 1/2/3 rings

        # Movement heads
        self.move_source = nn.Conv2d(embed_dim, 1, 1)  # spatial
        self.move_direction = nn.Linear(embed_dim, 8)  # 8 directions
        self.move_distance = nn.Linear(embed_dim, 20)  # buckets 1-20+

        # Capture heads (applied recursively)
        self.capture_continue = nn.Linear(embed_dim, 2)  # yes/no
        self.capture_target = nn.Conv2d(embed_dim, 1, 1)  # spatial

        # Post-processing heads
        self.line_choice = nn.Linear(embed_dim, MAX_LINES + 1)  # +1 for none
        self.line_option = nn.Linear(embed_dim, 2)
        self.region_choice = nn.Linear(embed_dim, MAX_REGIONS + 1)
```

### 2.3 Capture Chain Handling (Critical)

The capture chain is the most complex part. Two approaches:

**Approach A: Enumeration with Pruning**

```python
def enumerate_capture_chains(state, start_pos, max_chains=100):
    """BFS over capture sequences, pruning by heuristic value."""
    frontier = [(start_pos, state, [])]
    completed = []

    while frontier and len(completed) < max_chains:
        pos, current_state, chain = frontier.pop(0)
        legal_captures = get_legal_captures(current_state, pos)

        if not legal_captures:
            completed.append(chain)
            continue

        # Score and prune captures
        scored = [(c, capture_heuristic(current_state, c)) for c in legal_captures]
        scored.sort(key=lambda x: -x[1])
        top_captures = scored[:10]  # Keep top 10

        for capture, _ in top_captures:
            new_state = apply_capture(current_state, capture)
            new_pos = capture.landing_position
            frontier.append((new_pos, new_state, chain + [capture]))

    return completed
```

**Approach B: Learned Chain Policy (Recommended)**

```python
def sample_capture_chain(state, policy_net, start_pos):
    """Autoregressively sample captures using learned policy."""
    chain = []
    pos = start_pos

    while True:
        legal_captures = get_legal_captures(state, pos)
        if not legal_captures:
            break

        # Get policy distribution
        features = encode_capture_context(state, pos, chain)
        probs = policy_net.capture_head(features)

        # Mask illegal and sample
        masked_probs = mask_illegal(probs, legal_captures)
        capture = sample_from(masked_probs)

        if capture == STOP_TOKEN:
            break

        chain.append(capture)
        state = apply_capture(state, capture)
        pos = capture.landing_position

    return chain
```

---

## 3. Neural Network Architecture

### 3.1 Backbone Options

**Option A: ResNet (simpler, proven)**

```python
class ResNetBackbone(nn.Module):
    def __init__(self, in_channels=42, hidden=256, blocks=20):
        self.input_conv = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden) for _ in range(blocks)
        ])

    def forward(self, x):
        x = F.relu(self.input_conv(x))
        for block in self.blocks:
            x = block(x)
        return x
```

**Option B: Graph Attention Network (better for non-locality)**

```python
class GraphAttentionBackbone(nn.Module):
    def __init__(self, node_features=42, hidden=256, layers=8, heads=8):
        self.input_proj = nn.Linear(node_features, hidden)
        self.layers = nn.ModuleList([
            GraphAttentionLayer(hidden, heads) for _ in range(layers)
        ])

    def forward(self, node_features, adjacency):
        # adjacency includes: movement edges, territory edges, line edges
        x = self.input_proj(node_features)
        for layer in self.layers:
            x = layer(x, adjacency)
        return x
```

### 3.2 Value Head

```python
class ValueHead(nn.Module):
    def __init__(self, spatial_dim, hidden, num_players):
        self.conv = nn.Conv2d(hidden, 1, 1)
        self.fc1 = nn.Linear(spatial_dim, 256)
        self.fc2 = nn.Linear(256, num_players)  # Per-player win prob

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)  # Win probability per player
```

### 3.3 Auxiliary Heads (for training stability)

```python
class AuxiliaryHeads(nn.Module):
    """Predict game-relevant quantities to improve representation learning."""

    def __init__(self, hidden):
        # Ownership prediction (like KataGo)
        self.ownership = nn.Conv2d(hidden, 4, 1)  # Who will own each cell?

        # Territory prediction
        self.territory = nn.Linear(hidden, 4)  # Final territory per player

        # Elimination prediction
        self.eliminations = nn.Linear(hidden, 4)  # Rings eliminated per player

        # Line completion prediction
        self.line_probs = nn.Conv2d(hidden, 4, 1)  # Line completion likelihood
```

---

## 4. Search Algorithm

### 4.1 MCTS with Progressive Widening

```python
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = np.zeros(num_players)
        self.prior = None  # Set by policy network

    def expanded_fraction(self):
        return len(self.children) / self.max_children()

    def max_children(self):
        # Progressive widening: allow more children as visits increase
        return int(C_PW * (self.visit_count ** ALPHA_PW))
```

### 4.2 Tactical Oracle Integration

```python
class TacticalOracles:
    """Fast heuristic computations to guide search."""

    def capture_value(self, state, capture_chain):
        """Estimate value of a capture sequence."""
        material_gain = sum(c.rings_captured for c in capture_chain)
        position_value = self.position_heuristic(capture_chain[-1].landing)
        vulnerability = self.vulnerability_after(state, capture_chain)
        return material_gain + 0.3 * position_value - 0.5 * vulnerability

    def disconnection_threat(self, state, player):
        """Check if player has regions at risk of disconnection."""
        regions = self.find_player_regions(state, player)
        threats = []
        for region in regions:
            border_strength = self.border_strength(state, region)
            if border_strength < THRESHOLD:
                threats.append((region, border_strength))
        return threats

    def line_completion_moves(self, state, player):
        """Find moves that complete or extend lines."""
        lines = self.find_partial_lines(state, player)
        completions = []
        for line in lines:
            extending_moves = self.moves_extending_line(state, line)
            completions.extend(extending_moves)
        return completions
```

### 4.3 Multi-Player Search Modification

```python
def mcts_backup(node, result):
    """Backup with vector values for multiplayer."""
    while node is not None:
        node.visit_count += 1
        node.value_sum += result  # result is [p0_score, p1_score, p2_score, ...]
        node = node.parent

def select_action(node, player):
    """PUCT selection considering current player's value."""
    best_score = -float('inf')
    best_action = None

    for action, child in node.children.items():
        q_value = child.value_sum[player] / child.visit_count
        u_value = C_PUCT * child.prior * sqrt(node.visit_count) / (1 + child.visit_count)
        score = q_value + u_value

        if score > best_score:
            best_score = score
            best_action = action

    return best_action
```

---

## 5. Training Pipeline

### 5.1 Self-Play Data Generation

```python
def self_play_game(model, config):
    state = initial_state(config.board_type, config.num_players)
    trajectory = []

    while not state.is_terminal():
        # Run MCTS
        root = MCTSNode(state)
        for _ in range(config.simulations):
            mcts_simulate(root, model)

        # Extract policy from visit counts
        policy = {a: c.visit_count for a, c in root.children.items()}
        policy = normalize(policy)

        # Sample action (with temperature)
        if state.move_number < config.temp_threshold:
            action = sample_from(policy, temperature=config.temperature)
        else:
            action = argmax(policy)

        trajectory.append((state, policy, state.current_player))
        state = apply_action(state, action)

    # Assign rewards
    winner = state.winner()
    for state, policy, player in trajectory:
        if config.num_players == 2:
            reward = 1.0 if player == winner else -1.0
        else:
            reward = np.zeros(config.num_players)
            reward[winner] = 1.0
        yield (state, policy, reward)
```

### 5.2 Training Loss

```python
def training_loss(model, batch):
    states, target_policies, target_values = batch

    # Forward pass
    pred_policies, pred_values, aux_outputs = model(states)

    # Policy loss (cross-entropy)
    policy_loss = cross_entropy(pred_policies, target_policies)

    # Value loss (MSE for 2-player, cross-entropy for multiplayer)
    if num_players == 2:
        value_loss = mse_loss(pred_values, target_values)
    else:
        value_loss = cross_entropy(pred_values, target_values)

    # Auxiliary losses
    ownership_loss = cross_entropy(aux_outputs.ownership, compute_final_ownership(batch))

    # Total loss
    total = policy_loss + value_loss + 0.5 * ownership_loss + L2_REG * model.l2_norm()
    return total
```

### 5.3 Curriculum Strategy

```
Phase 1: 8×8 2-player (1 week)
├── Disable territory processing initially
├── Focus on movement, capture, basic lines
└── Target: Beat random baseline consistently

Phase 2: 8×8 2-player full rules (1 week)
├── Enable all rules
├── Learn territory and disconnection
└── Target: Beat heuristic AI

Phase 3: 19×19 2-player (2 weeks)
├── Transfer from 8×8 weights
├── Larger search budget
└── Target: Reasonable play quality

Phase 4: 3-player modes (ongoing)
├── Vector value head
├── Opponent modeling
└── Target: Handle kingmaking reasonably
```

---

## 6. Integration with Existing Codebase

### 6.1 Interface with Python Rules Engine

```python
# In ai-service/app/ai/neural_ai.py

class NeuralAI:
    def __init__(self, model_path, device='cuda'):
        self.model = load_model(model_path).to(device)
        self.model.eval()

    def select_move(self, state: GameState, time_budget_ms: int) -> Move:
        # Encode state
        board_tensor = encode_board(state)
        global_features = encode_global(state)

        # Run MCTS with time budget
        root = MCTSNode(state)
        simulations = estimate_simulations(time_budget_ms)

        for _ in range(simulations):
            self.mcts_simulate(root)

        # Select best action
        best_action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        return decode_action(best_action, state)

    def mcts_simulate(self, root):
        node = root
        path = [node]

        # Selection
        while node.is_expanded() and not node.state.is_terminal():
            action = select_action(node, node.state.current_player)
            node = node.children[action]
            path.append(node)

        # Expansion
        if not node.state.is_terminal():
            state_encoding = encode_state(node.state)
            policy, value = self.model(state_encoding)
            node.expand(policy)
        else:
            value = terminal_value(node.state)

        # Backup
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
```

### 6.2 Training Script Entry Points

```bash
# Export training data from selfplay games
python scripts/export_replay_dataset.py \
    --db data/games/cluster_merged.db \
    --output data/training_data/

# Train neural network baseline
python scripts/run_nn_training_baseline.py \
    --board square8 \
    --num-players 2 \
    --data-dir data/training_data/ \
    --output models/neural_v1

# Train NNUE-style network
python scripts/train_nnue.py \
    --board square8 \
    --num-players 2 \
    --checkpoint models/neural_v1/best.pt \
    --output models/nnue_v1
```

> **Note:** The legacy `app.training.train_neural` module has been replaced by the scripts above. See `config/unified_loop.yaml` for the canonical training configuration.

---

## 7. Comparison: Neural AI vs Current Heuristic AI

| Aspect                | Current Heuristic AI           | Proposed Neural AI               |
| --------------------- | ------------------------------ | -------------------------------- |
| **Evaluation**        | Hand-tuned weights (45 params) | Learned (millions of params)     |
| **Search**            | 1-ply with move ordering       | MCTS (hundreds of playouts)      |
| **Capture handling**  | Best-first heuristic           | Learned chain policy             |
| **Territory**         | Static heuristics              | Ownership prediction head        |
| **Multiplayer**       | Max-N approximation            | Vector value head                |
| **Training**          | CMA-ES (hours)                 | Self-play (days/weeks)           |
| **Expected strength** | Intermediate                   | Strong (if trained sufficiently) |

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Implement board tensor encoding
- [ ] Implement global feature extraction
- [ ] Create ResNet backbone (simple version)
- [ ] Single policy head for move selection (no capture chains yet)
- [ ] Basic value head (2-player scalar)

### Phase 2: Action Decomposition (Week 3-4)

- [ ] Factored policy heads
- [ ] Capture chain enumeration with pruning
- [ ] Integration with existing `GameEngine.apply_move()`

### Phase 3: Training Infrastructure (Week 5-6)

- [ ] Self-play game generation
- [ ] Training loop with replay buffer
- [ ] Checkpointing and evaluation

### Phase 4: Search and Optimization (Week 7-8)

- [ ] Full MCTS implementation
- [ ] Progressive widening
- [ ] Tactical oracles integration

### Phase 5: Multiplayer and Refinement (Ongoing)

- [ ] Vector value head for 3-4 players
- [ ] Auxiliary training targets
- [ ] Curriculum learning across board sizes

---

## 9. Implemented Architectures (2025-12)

### 9.1 HexNeuralNet Variants

Specialized architectures for hexagonal boards are implemented in `app/ai/neural_net.py`.

#### HexNeuralNet_v2

```python
# Location: neural_net.py:5519
class HexNeuralNet_v2:
    """Original hex architecture with 10 channels per player."""

    def __init__(self, board_size=25, policy_size=91876, in_channels=40):
        # 40 channels = 10 channels × 4 players
        # Uses HexStateEncoderV2
```

#### HexNeuralNet_v3 (Recommended)

```python
# Location: neural_net.py:5816
class HexNeuralNet_v3:
    """Improved hex architecture with 16 channels per player."""

    def __init__(self, board_size=25, policy_size=91876, in_channels=64):
        # 64 channels = 16 channels × 4 players
        # Uses HexStateEncoderV3 (recommended)
```

**Configuration** (`unified_loop.yaml`):

```yaml
training:
  hex_encoder_version: 'v3' # Select encoder version
```

#### Supported Board Sizes

| Board Type | Bounding Box | Policy Size | Encoder           |
| ---------- | ------------ | ----------- | ----------------- |
| hex8       | 9×9          | ~4,500      | HexStateEncoderV3 |
| hexagonal  | 25×25        | ~92,000     | HexStateEncoderV3 |

### 9.2 Gumbel MCTS

Efficient search algorithm using Gumbel-Top-K sampling with Sequential Halving.

**Location**: `app/ai/gumbel_mcts_ai.py`

```python
class GumbelMCTSAI:
    """
    Gumbel MCTS with Sequential Halving.

    Key features:
    - Gumbel noise for diverse action sampling
    - Sequential Halving for efficient simulation allocation
    - Completed Q-values for visit asymmetry
    - Visit distribution extraction for soft policy targets
    """

    def select_move(self, game_state) -> Move:
        """Select best move using Gumbel MCTS."""

    def get_visit_distribution(self) -> Tuple[List[Move], List[float]]:
        """Extract normalized visit counts for training."""
```

**Configuration**:

```python
config = AIConfig(
    ai_type=AIType.GUMBEL_MCTS,
    gumbel_num_sampled_actions=16,  # Actions to sample
    gumbel_simulation_budget=100,   # Total simulations
    use_neural_net=True,
)
```

**Usage in Selfplay**:

```bash
python scripts/run_hybrid_selfplay.py \
  --board-type hex8 \
  --engine-mode gumbel-mcts \
  --nn-model-id ringrift_hex8_2p_v3_retrained
```

### 9.3 HexStateEncoderV3

State encoder producing 16 channels per player for hex boards.

**Location**: `app/training/encoding.py`

```python
class HexStateEncoderV3:
    """
    Hex state encoder with 16 channels per frame.

    Channels per player:
    - 4 channels: Stack ownership
    - 4 channels: Stack height buckets
    - 4 channels: Ring placement
    - 4 channels: Territory control
    """

    def __init__(self, board_size=25, policy_size=91876):
        self.channels_per_frame = 16
        # With 4-frame history: 64 total channels
```

---

## 10. Related Documentation

- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Training configuration reference
- [GUMBEL_MCTS.md](GUMBEL_MCTS.md) - Gumbel MCTS implementation details
- [HEX_AUGMENTATION.md](HEX_AUGMENTATION.md) - D6 symmetry augmentation

---

_Document created: December 2025_
_Last updated: 2025-12-17_
_Based on analysis of AlphaZero, AlphaStar, KataGo, and Stockfish NNUE architectures_
