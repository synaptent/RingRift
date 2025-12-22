#!/bin/bash
# Monitor Gumbel MCTS selfplay cluster - 22 Lambda GH200 nodes
# Quick status check for morning review

set -e

check_node() {
    local ip=$1
    local board=$2
    local players=$3

    # Check process and game count in one SSH call
    result=$(ssh -o ConnectTimeout=10 -o BatchMode=yes ubuntu@${ip} "
        pid=\$(pgrep -f generate_gumbel_selfplay 2>/dev/null | head -1)
        games=\$(wc -l ~/ringrift/ai-service/data/selfplay/gumbel_${board}_${players}p.jsonl 2>/dev/null | awk '{print \$1}')
        echo \"\${pid:-0}|\${games:-0}\"
    " 2>/dev/null || echo "0|0")

    pid=$(echo "$result" | cut -d'|' -f1)
    games=$(echo "$result" | cut -d'|' -f2)

    if [ "$pid" != "0" ] && [ -n "$pid" ]; then
        status="✓"
    else
        status="✗"
    fi

    printf "  %-18s %-10s %s %6s games\n" "$ip" "($board)" "$status" "$games"
}

echo "=============================================="
echo "  GUMBEL MCTS CLUSTER STATUS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

echo ""
echo "=== 2P NODES (7 nodes) ==="
check_node "100.65.88.62" "square8" 2
check_node "100.79.109.120" "square8" 2
check_node "100.117.177.83" "square19" 2
check_node "100.99.27.56" "square19" 2
check_node "100.66.65.33" "hexagonal" 2
check_node "100.104.126.58" "hex8" 2
check_node "100.83.234.82" "hex8" 2

echo ""
echo "=== 3P NODES (8 nodes) ==="
check_node "100.123.183.70" "square8" 3
check_node "100.88.35.19" "square8" 3
check_node "100.75.84.47" "square19" 3
check_node "100.88.176.74" "square19" 3
check_node "100.104.165.116" "hexagonal" 3
check_node "100.96.142.42" "hexagonal" 3
check_node "100.76.145.60" "hex8" 3
check_node "100.85.106.113" "hex8" 3

echo ""
echo "=== 4P NODES (7 nodes) ==="
check_node "100.106.0.3" "square8" 4
check_node "100.81.5.33" "square8" 4
check_node "100.78.101.123" "square19" 4
check_node "100.97.104.89" "square19" 4
check_node "100.91.25.13" "hexagonal" 4
check_node "100.101.45.4" "hexagonal" 4
check_node "100.78.55.103" "hex8" 4

echo ""
echo "=============================================="
echo "  AGGREGATED GAME COUNTS (by board)"
echo "=============================================="

echo ""
printf "%-12s %8s %8s %8s %10s\n" "Board" "2P" "3P" "4P" "Total"
printf "%-12s %8s %8s %8s %10s\n" "--------" "------" "------" "------" "--------"

total_games_2p=0
total_games_3p=0
total_games_4p=0

for board in square8 square19 hexagonal hex8; do
    # Get counts from representative nodes
    case $board in
        square8) ip_2p="100.65.88.62"; ip_3p="100.123.183.70"; ip_4p="100.106.0.3" ;;
        square19) ip_2p="100.117.177.83"; ip_3p="100.75.84.47"; ip_4p="100.78.101.123" ;;
        hexagonal) ip_2p="100.66.65.33"; ip_3p="100.104.165.116"; ip_4p="100.91.25.13" ;;
        hex8) ip_2p="100.104.126.58"; ip_3p="100.76.145.60"; ip_4p="100.78.55.103" ;;
    esac

    c2=$(ssh -o ConnectTimeout=5 ubuntu@$ip_2p "wc -l ~/ringrift/ai-service/data/selfplay/gumbel_${board}_2p.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo 0)
    c3=$(ssh -o ConnectTimeout=5 ubuntu@$ip_3p "wc -l ~/ringrift/ai-service/data/selfplay/gumbel_${board}_3p.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo 0)
    c4=$(ssh -o ConnectTimeout=5 ubuntu@$ip_4p "wc -l ~/ringrift/ai-service/data/selfplay/gumbel_${board}_4p.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo 0)

    [ -z "$c2" ] && c2=0
    [ -z "$c3" ] && c3=0
    [ -z "$c4" ] && c4=0

    total=$((c2 + c3 + c4))
    printf "%-12s %8d %8d %8d %10d\n" "$board" "$c2" "$c3" "$c4" "$total"

    total_games_2p=$((total_games_2p + c2))
    total_games_3p=$((total_games_3p + c3))
    total_games_4p=$((total_games_4p + c4))
done

grand_total=$((total_games_2p + total_games_3p + total_games_4p))
printf "%-12s %8s %8s %8s %10s\n" "--------" "------" "------" "------" "--------"
printf "%-12s %8d %8d %8d %10d\n" "TOTAL" "$total_games_2p" "$total_games_3p" "$total_games_4p" "$grand_total"

echo ""
echo "=============================================="
