import sys
import os
import time
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.training.generate_data import generate_dataset
from app.training.train import train_model
from app.models import AIConfig
from app.ai.mcts_ai import MCTSAI
from app.ai.neural_net import NeuralNetAI

def run_training_loop(iterations=5, games_per_iter=10, epochs_per_iter=5):
    print(f"Starting training loop: {iterations} iterations, {games_per_iter} games/iter")
    
    # Use absolute paths
    base_dir = os.getcwd()
    data_file = os.path.join(base_dir, "app/training/data/self_play_data.npy")
    model_file = os.path.join(base_dir, "app/models/ringrift_v1.pth")
    
    for i in range(iterations):
        print(f"\n=== Iteration {i+1}/{iterations} ===")
        
        # 1. Self-Play (MCTS vs MCTS)
        # Note: generate_dataset currently uses Heuristic vs Minimax.
        # We should update it or override it to use MCTS vs MCTS (guided by current NN).
        # But generate_dataset hardcodes the AIs.
        # I will create a custom generation function here.
        
        print("Generating self-play data...")
        
        # Initialize MCTS AIs
        # They will use the current neural net (if available) for evaluation
        ai1 = MCTSAI(1, AIConfig(difficulty=5))
        ai2 = MCTSAI(2, AIConfig(difficulty=5))
        
        # Generate data
        generate_dataset(num_games=games_per_iter, output_file=data_file, ai1=ai1, ai2=ai2)
        
        # 2. Train Neural Net
        print("Training neural network...")
        train_model(data_path=data_file, save_path=model_file, epochs=epochs_per_iter)
        
        # 3. Evaluation (Optional)
        # We could run a tournament here to see if the new model is better
        print("Iteration complete.")

if __name__ == "__main__":
    run_training_loop()