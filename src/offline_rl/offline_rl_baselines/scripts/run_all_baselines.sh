#!/bin/bash
# 批量运行所有baseline实验
# 使用方法: bash run_all_baselines.sh [algorithm]
# 示例: bash run_all_baselines.sh td3_bc  # 只运行TD3+BC
#       bash run_all_baselines.sh all      # 运行所有算法（默认）

# 项目根目录
PROJECT_ROOT="/data/liyuefeng/gems/gems_official/official_code"
SCRIPT_DIR="${PROJECT_ROOT}/offline_rl_baselines/scripts"
LOG_DIR="${PROJECT_ROOT}/offline_rl_baselines/experiments/logs"

# 确保log目录存在
mkdir -p ${LOG_DIR}

# 获取当前日期时间
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "开始批量运行离线RL Baseline实验"
echo "时间戳: ${TIMESTAMP}"
echo "=========================================="
echo ""

# 环境列表
ENVS=("diffuse_topdown" "diffuse_mix" "diffuse_divpen")

# 算法列表
# 注意：CQL和IQL已移植但需要进一步适配训练脚本
# 当前完全可用：td3_bc
# 待完善：cql, iql
if [ "$1" == "td3_bc" ]; then
    ALGOS=("td3_bc")
    echo "只运行 TD3+BC"
elif [ "$1" == "cql" ]; then
    ALGOS=("cql")
    echo "只运行 CQL (注意：需要先完善训练脚本)"
elif [ "$1" == "iql" ]; then
    ALGOS=("iql")
    echo "只运行 IQL (注意：需要先完善训练脚本)"
else
    ALGOS=("td3_bc")
    echo "默认运行 TD3+BC"
    echo "提示：CQL和IQL算法文件已准备好，但训练脚本需要进一步完善"
fi

# Seeds
SEEDS=(0 1 2)

# 遍历所有组合
for env in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            exp_name="${algo}_${env}_seed${seed}"

            echo ""
            echo "启动实验: ${exp_name}"
            echo "  环境: ${env}"
            echo "  算法: ${algo}"
            echo "  种子: ${seed}"

            # 运行训练
            cd ${PROJECT_ROOT}
            nohup python ${SCRIPT_DIR}/train_${algo}.py \
                --env_name ${env} \
                --seed ${seed} \
                --max_timesteps 1000000 \
                --batch_size 256 \
                --alpha 2.5 \
                --normalize \
                --device cuda \
                > ${LOG_DIR}/${exp_name}_${TIMESTAMP}.log 2>&1 &

            PID=$!
            echo "  PID: ${PID}"
            echo "  日志: ${LOG_DIR}/${exp_name}_${TIMESTAMP}.log"

            # 避免同时启动太多任务
            sleep 5
        done
    done
done

echo ""
echo "=========================================="
echo "所有实验已启动"
echo "=========================================="
echo ""
echo "查看日志:"
echo "  ls ${LOG_DIR}/"
echo ""
echo "查看进程:"
echo "  ps aux | grep train_"
echo ""
echo "停止所有任务:"
echo "  pkill -f train_td3_bc.py"
echo ""
