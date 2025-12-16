#!/bin/bash
cd ~/ringrift/ai-service
LOG=/tmp/aggregation.log
echo "[$(date)] Starting aggregation" >> $LOG
venv/bin/python scripts/aggregate_jsonl_to_db.py --input-dir data/selfplay --output data/games/jsonl_aggregated.db >> $LOG 2>&1
echo "[$(date)] Done. Games: $(sqlite3 data/games/jsonl_aggregated.db "SELECT COUNT(*) FROM games" 2>/dev/null)" >> $LOG
