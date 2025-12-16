#!/bin/bash
# Deploy decentralized operations to all cluster nodes
echo "=== Deploying decentralization changes ==="
for host in lambda-gh200-a lambda-gh200-b lambda-gh200-c lambda-gh200-d lambda-gh200-e lambda-gh200-f lambda-gh200-g lambda-gh200-h lambda-gh200-i lambda-gh200-k lambda-gh200-l lambda-2xh100 lambda-h100 lambda-a10; do
  echo "--- $host ---"
  timeout 30 ssh "$host" 'cd ~/ringrift && git pull origin main 2>&1 | tail -1 && sudo systemctl restart ringrift-p2p && echo "Restarted"' 2>/dev/null || echo "$host: TIMEOUT/ERROR"
done
echo "=== Deployment complete ==="
