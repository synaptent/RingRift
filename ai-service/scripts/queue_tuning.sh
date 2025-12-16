#\!/bin/bash
cd ~/ringrift/ai-service
source venv/bin/activate 2>/dev/null || true

echo "=== Queued tuning started at $(date) ===" >> logs/hp_tuning/queue.log

# Wait for square8_3p to finish
echo "Waiting for square8_3p to complete..." >> logs/hp_tuning/queue.log
while pgrep -f "tune_hyperparameters.*square8.*players 3" > /dev/null 2>&1; do
    sleep 60
done
echo "square8_3p completed at $(date)" >> logs/hp_tuning/queue.log

# Run square19_3p
echo "Starting square19_3p at $(date)" >> logs/hp_tuning/queue.log
python3 scripts/tune_hyperparameters.py --board square19 --players 3 --trials 30 --epochs 15 --db data/games/selfplay.db --output-dir logs/hp_tuning/square19_3p > logs/hp_tuning/square19_3p.log 2>&1
echo "square19_3p completed at $(date)" >> logs/hp_tuning/queue.log

# Run hexagonal_3p
echo "Starting hexagonal_3p at $(date)" >> logs/hp_tuning/queue.log
python3 scripts/tune_hyperparameters.py --board hexagonal --players 3 --trials 30 --epochs 15 --db data/games/selfplay.db --output-dir logs/hp_tuning/hexagonal_3p > logs/hp_tuning/hexagonal_3p.log 2>&1
echo "hexagonal_3p completed at $(date)" >> logs/hp_tuning/queue.log

echo "=== All tuning complete on this host at $(date) ===" >> logs/hp_tuning/queue.log
