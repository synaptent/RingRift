#!/bin/bash
for host in lambda-gh200-e lambda-gh200-c lambda-gh200-a lambda-h100 lambda-2xh100 lambda-a10; do
  result=$(timeout 8 ssh "$host" 'curl -s --connect-timeout 5 http://localhost:8770/status 2>/dev/null')
  if [ -n "$result" ]; then
    echo "$result" | jq -c '{n:.node_id, r:.role, l:.leader_id, ap:.alive_peers, vq:.voter_quorum_ok}'
  else
    echo "{\"n\":\"$host\",\"r\":\"OFFLINE\"}"
  fi
done
