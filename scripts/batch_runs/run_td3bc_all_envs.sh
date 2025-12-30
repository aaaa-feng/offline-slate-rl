#!/bin/bash

################################################################################
# TD3+BC 批量运行脚本 - 所有6个环境
# 用法: bash scripts/run_td3bc_all_envs.sh [seed] [alpha] [gamma]
# 示例: bash scripts/run_td3bc_all_envs.sh 58407201 5.0 0.95
################################################################################

# 默认参数
SEED=${1:-58407201}
ALPHA=${2:-5.0}
GAMMA=${3:-0.95}

echo "================================================================================"
echo "=== TD3+BC Batch Training - All 6 Environments ==="
echo "================================================================================"
echo "Seed: ${SEED}"
echo "Alpha: ${ALPHA}"
echo "Gamma: ${GAMMA}"
echo "================================================================================"
echo ""

# 6个环境列表
ENVS=("diffuse_topdown" "diffuse_mix" "diffuse_divpen" "focused_topdown" "focused_mix" "focused_divpen")

# 依次运行每个环境
for env in "${ENVS[@]}"; do
    echo "🚀 Starting training for ${env}..."
    bash scripts/run_td3bc_nohup.sh ${env} ${SEED} ${ALPHA} ${GAMMA}
    echo "✅ ${env} training started"
    echo ""
    sleep 2  # 等待2秒,避免同时启动太多进程
done

echo "================================================================================"
echo "✅ All 6 environments training started!"
echo "================================================================================"
echo ""
echo "📊 Monitor all trainings:"
echo "  ps aux | grep td3_bc_v2.py"
echo ""
echo "📁 Log files location:"
echo "  /data/liyuefeng/offline-slate-rl/experiments/logs/offline/td3_bc/"
echo ""
