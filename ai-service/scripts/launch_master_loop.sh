#!/bin/bash
# Launch wrapper for master_loop.py — used by com.ringrift.master-loop LaunchAgent.
# Uses .venv python (3.11) instead of system python (3.9) which can't import the codebase.
export PATH="/opt/homebrew/bin:$PATH"
cd /Users/armand/Development/RingRift/ai-service
export PYTHONPATH=.
exec .venv/bin/python3 scripts/master_loop.py
