#!/bin/bash

################################################################################
# TD3+BC 多seed运行脚本
# 用法: bash scripts/run_td3bc_multi_seeds.sh <env_name> [alpha] [gamma]
# 示例: bash scripts/run_td3bc_multi_seeds.sh diffuse_topdown 5.0 0.95
################################################################################

# 检查参数
if [ "$#" -lt 1 ]; then
    echo "错误: 需要至少1个参数"
    echo "用法: bash scripts/run_td3bc_multi_seeds.sh <env_name> [alpha] [gamma]"
    echo "示例: bash scripts/run_td3bc_multi_seeds.sh diffuse_topdown 5.0 0.95"
    exit 1
fi

ENV_NAME=$1
ALPHA=${2:-5.0}
GAMMA=${3:-0.95}

# 5个连续的seed (与在线算法一致)
SEEDS=(58407201 58407202 58407203 58407204 58407205)

echo "================================================================================"
echo "=== TD3+BC Multi-Seed Training ==="
echo "================================================================================"
echo "Environment: ${ENV_NAME}"
echo "Alpha: ${ALPHA}"
echo "Gamma: ${GAMMA}"
echo "Seeds: ${SEEDS[@]}"
echo "================================================================================"
echo ""

# 依次运行每个seed
for seed in "${SEEDS[@]}"; do
    echo "🚀 Starting training with seed ${seed}..."
    bash scripts/run_td3bc_nohup.sh ${ENV_NAME} ${seed} ${ALPHA} ${GAMMA}
    echo "✅ Seed ${seed} training started"
    echo ""
    sleep 2  # 等待2秒
done

echo "================================================================================"
echo "✅ All 5 seeds training started for ${ENV_NAME}!"
echo "================================================================================"
echo ""
echo "📊 Monitor all trainings:"
echo "  ps aux | grep td3_bc_v2.py | grep ${ENV_NAME}"
echo ""
