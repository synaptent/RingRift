#!/bin/bash
# P2P Recovery Cron Script
# Runs every 10 minutes to restart dead P2P orchestrators

cd /Users/armand/Development/RingRift/ai-service

# Log file for cron output
LOG=/tmp/p2p_recovery.log

echo "$(date): Running P2P recovery check" >> $LOG

# Start P2P on any offline instances (skip already running)
python scripts/vast_p2p_manager.py start --skip-running >> $LOG 2>&1

echo "$(date): Recovery check complete" >> $LOG
