#!/bin/bash
export PATH=/usr/sbin:/usr/bin:/bin:/sbin:/opt/homebrew/bin:$PATH
export PYTHONPATH=/Users/armand/Development/RingRift/ai-service
cd /Users/armand/Development/RingRift/ai-service
exec /Users/armand/.pyenv/versions/3.10.13/bin/python3 scripts/p2p_orchestrator.py \
  --node-id macbook-pro-5 \
  --port 8770 \
  --peers http://100.107.168.125:8770,http://100.94.201.92:8770,http://100.94.174.19:8770 \
  --ringrift-path /Users/armand/Development/RingRift
