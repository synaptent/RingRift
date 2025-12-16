#!/bin/bash
cd /Users/armand/Development/RingRift/ai-service
PYTHONPATH=. /Users/armand/.pyenv/versions/3.10.13/bin/python3 scripts/sync_models.py --sync >> logs/model_sync_cron.log 2>&1
