import os
import numpy as np


def reanalyze_dataset(
    input_file="data/dataset.npz",
    output_file="data/dataset_reanalyzed.npz",
    model_path="ai-service/app/models/ringrift_best.pth"
):
    """
    Re-analyze a dataset using the latest model and Descent search.
    Generates fresh targets (best_move, root_value) for existing states.

    NOTE: This is a stub implementation. Full ReAnalyze requires raw GameStates,
    not just feature tensors. The current dataset format only stores features,
    which are lossy and cannot be reversed to reconstruct full game state.

    To implement proper ReAnalyze:
    1. Save raw GameStates (pickled) or replay logs during data generation
    2. Load states, run DescentAI search to get fresh policy/value targets
    3. Re-extract features and save updated dataset

    Args:
        input_file: Path to input dataset (npz format)
        output_file: Path to output dataset (npz format)
        model_path: Path to model weights for search
    """
    # Suppress unused parameter warnings
    _ = output_file
    _ = model_path

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    print(f"Checking {input_file} format...")

    # Load data
    data = np.load(input_file, allow_pickle=True)

    # Handle both new and old formats
    if 'features' in data:
        # Feature-only format cannot be re-analyzed
        # We would need full GameState to run search

        # "Simple AlphaZero" ReAnalyze usually works on stored trajectories (states).
        # If we only have features, we can't easily reconstruct the full GameState
        # (stacks, markers, etc.) perfectly unless the features are lossless.
        # Our features are likely lossy or at least hard to reverse.

        # For this prototype, we can't implement full ReAnalyze on the *feature* dataset.
        # We would need to save the raw GameStates (pickled) or a replay log.

        # However, the plan said: "Loads saved games (states) from disk."
        # So we should assume we have a way to load states.
        # Let's assume for now we can't do it on the .npz file.

        print("Error: Cannot re-analyze from features only. Need raw GameStates.")
        return
    else:
        # Old format might have been different, but likely same issue.
        pass

    # To support ReAnalyze properly, we should modify generate_data to save
    # raw GameStates or move lists.
    # For now, I will implement the structure but note the limitation.

    # If we had a list of GameStates:
    # states = load_states(input_file)

    # ai = DescentAI(1, AIConfig(difficulty=10, think_time=500))
    # # Load model
    # if os.path.exists(model_path):
    #    ai.neural_net.model.load_state_dict(torch.load(model_path))

    # new_policies = []
    # new_values = []

    # for state in states:
    #     # Run search
    #     best_move = ai.select_move(state)
    #     root_value = ai.transposition_table[ai._get_state_key(state)][0]
    #
    #     # Encode
    #     policy = np.zeros(...)
    #     idx = ai.neural_net.encode_move(best_move, state.board.size)
    #     policy[idx] = 1.0
    #
    #     new_policies.append(policy)
    #     new_values.append(root_value)

    # Save new dataset
    # np.savez(output_file, features=features, globals=globals_vec, values=new_values, policies=new_policies)

    print("ReAnalyze implementation pending full state storage.")

if __name__ == "__main__":
    reanalyze_dataset()
