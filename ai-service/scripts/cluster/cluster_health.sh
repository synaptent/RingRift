#!/bin/bash

check_host() {
  local ip=$1
  local name=$2
  local user=$3
  
  result=$(ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no ${user}@${ip} '
    hostname=$(hostname)
    uptime_str=$(uptime | sed "s/.*load average: //" | cut -d, -f1)
    
    # CPU/Memory
    if command -v free &>/dev/null; then
      mem=$(free -m | awk "/Mem:/ {printf \"%.0f\", \$3/\$2*100}")
    else
      mem=$(vm_stat 2>/dev/null | awk "/Pages active/ {print int(\$3/256)}" || echo "N/A")
    fi
    
    # GPU if available
    gpu="N/A"
    if command -v nvidia-smi &>/dev/null; then
      gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d " ")
    fi
    
    # Process counts
    selfplay=$(pgrep -fc "selfplay|self_play" 2>/dev/null || echo 0)
    training=$(pgrep -fc "train" 2>/dev/null || echo 0)
    python_total=$(pgrep -fc python 2>/dev/null || echo 0)
    
    # Game count (quick check)
    games=0
    for db in ~/ringrift/ai-service/data/games/*.db ~/Development/RingRift/ai-service/data/games/*.db 2>/dev/null; do
      if [ -f "$db" ]; then
        c=$(sqlite3 "$db" "SELECT COUNT(*) FROM games" 2>/dev/null || echo 0)
        games=$((games + c))
      fi
    done
    
    echo "${hostname}|${uptime_str}|${mem}%|${gpu}|${selfplay}|${training}|${python_total}|${games}"
  ' 2>/dev/null)
  
  if [ -n "$result" ]; then
    echo "$name|$result"
  else
    echo "$name|OFFLINE|-|-|-|-|-|-|-"
  fi
}

echo "NAME|HOSTNAME|LOAD|MEM|GPU|SELFPLAY|TRAIN|PYTHON|GAMES"
echo "----+--------+----+---+---+--------+-----+------+-----"

# GH200 nodes
check_host 100.88.176.74 "GH200-e" ubuntu &
check_host 100.104.165.116 "GH200-f" ubuntu &
check_host 100.104.126.58 "GH200-g" ubuntu &
check_host 100.65.88.62 "GH200-h" ubuntu &

# Lambda nodes (192.222.x.x)
check_host 100.123.183.70 "Lambda-a" ubuntu &
check_host 100.104.34.73 "Lambda-b" ubuntu &
check_host 100.96.142.42 "Lambda-c" ubuntu &
check_host 100.88.35.19 "Lambda-d" ubuntu &
check_host 100.76.145.60 "Lambda-e" ubuntu &
check_host 100.99.27.56 "Lambda-f" ubuntu &
check_host 100.105.66.41 "Lambda-g" ubuntu &

# AWS
check_host 100.121.198.28 "AWS-1" ubuntu &
check_host 100.115.97.24 "AWS-2" ubuntu &

# Vast
check_host 100.74.154.36 "Vast-3070" root &
check_host 100.118.201.85 "Vast-512cpu" root &
check_host 100.100.242.64 "Vast-4060ti" root &

# Mac
check_host 100.107.168.125 "Mac-Studio" armand &

wait
