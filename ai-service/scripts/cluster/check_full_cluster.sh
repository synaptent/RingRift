#!/bin/bash
for host in lambda-gh200-a lambda-gh200-b lambda-gh200-c lambda-gh200-d lambda-gh200-e lambda-gh200-f lambda-gh200-g lambda-gh200-h lambda-gh200-i lambda-gh200-k lambda-gh200-l lambda-2xh100 lambda-h100 lambda-a10; do
  result=$(timeout 8 ssh "$host" 'curl -s --connect-timeout 5 http://localhost:8770/health 2>/dev/null')
  if [ -n "$result" ]; then
    jobs=$(echo "$result" | jq -r '.selfplay_jobs // 0')
    train=$(echo "$result" | jq -r '.training_jobs // 0')
    role=$(echo "$result" | jq -r '.role // "?"')
    printf "%-18s [%s] jobs=%3d train=%d\n" "$host" "${role:0:1}" "$jobs" "$train"
  else
    printf "%-18s OFFLINE\n" "$host"
  fi
done
