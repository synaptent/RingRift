import os
import sys
import numpy as np
import torch
from app.ai.descent_ai import DescentAI
from app.models import AIConfig, GameState
from app.ai.neural_net import INVALID_MOVE_INDEX

def reanalyze_dataset(
    input_file="data/dataset.npz",
    output_file="data/dataset_reanalyzed.npz",
    model_path="ai-service/app/models/ringrift_best.pth"
):
    """
    Re-analyze a dataset using the latest model and Descent search.
    Generates fresh targets (best_move, root_value) for existing states.
    """
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    print(f"Re-analyzing {input_file} with model {model_path}...")

    # Load data
    data = np.load(input_file, allow_pickle=True)
    
    # Handle both new and old formats
    if 'features' in data:
        features = data['features']
        globals_vec = data['globals']
        # We don't need old values/policies, we'll generate new ones
        # But we need to reconstruct GameStates to run search?
        # Wait, DescentAI needs a GameState to run search.
        # Our dataset only stores features.
        # This is a problem for ReAnalyze if we don't store the full state.
        
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