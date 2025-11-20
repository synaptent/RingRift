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
from app.training.tournament import Tournament

def run_training_loop(iterations=5, games_per_iter=10, epochs_per_iter=5):
    print(f"Starting training loop: {iterations} iterations, {games_per_iter} games/iter")
    
    # Use absolute paths
    base_dir = os.getcwd()
    data_file = os.path.join(base_dir, "app/training/data/self_play_data.npy")
    model_file = os.path.join(base_dir, "app/models/ringrift_v1.pth")
    best_model_file = os.path.join(base_dir, "app/models/ringrift_best.pth")
    
    # Initialize best model if not exists
    if not os.path.exists(best_model_file) and os.path.exists(model_file):
        import shutil
        shutil.copy(model_file, best_model_file)
    
    for i in range(iterations):
        print(f"\n=== Iteration {i+1}/{iterations} ===")
        
        # 1. Self-Play (MCTS vs MCTS)
        print("Generating self-play data...")
        
        # Initialize MCTS AIs
        # They will use the current neural net (if available) for evaluation
        ai1 = MCTSAI(1, AIConfig(difficulty=5))
        ai2 = MCTSAI(2, AIConfig(difficulty=5))
        
        # Generate data
        generate_dataset(num_games=games_per_iter, output_file=data_file, ai1=ai1, ai2=ai2)
        
        # 2. Train Neural Net
        print("Training neural network...")
        # Train on current data, saving to candidate model file
        candidate_model_file = model_file.replace(".pth", "_candidate.pth")
        train_model(data_path=data_file, save_path=candidate_model_file, epochs=epochs_per_iter)
        
        # 3. Evaluation (Tournament)
        if os.path.exists(best_model_file):
            print("Running tournament: Candidate vs Best...")
            tournament = Tournament(candidate_model_file, best_model_file, num_games=10)
            results = tournament.run()
            
            # Promotion logic: Candidate must win > 55% of games (excluding draws)
            total_decisive = results["A"] + results["B"]
            if total_decisive > 0:
                win_rate = results["A"] / total_decisive
                print(f"Candidate win rate: {win_rate:.2f}")
                
                if win_rate > 0.55:
                    print("Candidate promoted to Best Model!")
                    import shutil
                    shutil.copy(candidate_model_file, best_model_file)
                    # Also update the main model file used for inference
                    shutil.copy(candidate_model_file, model_file)
                else:
                    print("Candidate failed to beat Best Model.")
            else:
                print("Tournament inconclusive (all draws). Keeping Best Model.")
        else:
            # First iteration, promote candidate immediately
            print("First model generated. Promoting to Best Model.")
            import shutil
            shutil.copy(candidate_model_file, best_model_file)
            shutil.copy(candidate_model_file, model_file)
            
        print("Iteration complete.")

if __name__ == "__main__":
    run_training_loop()