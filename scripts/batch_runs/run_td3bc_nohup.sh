#!/bin/bash

################################################################################
# TD3+BC Nohup训练脚本
# 用法: bash scripts/run_td3bc_nohup.sh <env_name> <seed> <alpha> <gamma>
# 示例: bash scripts/run_td3bc_nohup.sh diffuse_topdown 58407201 5.0 0.95
################################################################################

# 检查参数
if [ "$#" -ne 4 ]; then
    echo "错误: 需要4个参数"
    echo "用法: bash scripts/run_td3bc_nohup.sh <env_name> <seed> <alpha> <gamma>"
    echo "示例: bash scripts/run_td3bc_nohup.sh diffuse_topdown 58407201 5.0 0.95"
    exit 1
fi

ENV_NAME=$1
SEED=$2
ALPHA=$3
GAMMA=$4

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d")

# 项目根目录
PROJECT_ROOT="/data/liyuefeng/offline-slate-rl"

# 日志文件名 (单文件方案)
LOG_FILENAME="td3_bc_${ENV_NAME}_seed${SEED}_alpha${ALPHA}_gamma${GAMMA}_${TIMESTAMP}.log"

# 日志目录
LOG_DIR="${PROJECT_ROOT}/experiments/logs/offline/td3_bc/${ENV_NAME}"
mkdir -p ${LOG_DIR}

# 日志文件路径
LOG_FILE="${LOG_DIR}/${LOG_FILENAME}"

# Checkpoint目录
CKPT_DIR="${PROJECT_ROOT}/checkpoints/offline_rl/td3_bc/${ENV_NAME}"
mkdir -p ${CKPT_DIR}

# 数据集路径
DATASET_PATH="${PROJECT_ROOT}/data/datasets/offline/${ENV_NAME}/expert_data_d4rl.npz"

# 检查数据集是否存在
if [ ! -f "${DATASET_PATH}" ]; then
    echo "错误: 数据集不存在: ${DATASET_PATH}"
    exit 1
fi

# 打印配置信息
echo "================================================================================"
echo "=== TD3+BC Training Configuration ==="
echo "================================================================================"
echo "Environment: ${ENV_NAME}"
echo "Seed: ${SEED}"
echo "Alpha (BC weight): ${ALPHA}"
echo "Gamma (discount): ${GAMMA}"
echo "Dataset: ${DATASET_PATH}"
echo "Log file: ${LOG_FILE}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "================================================================================"
echo ""

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gems

# 运行训练 (使用python -u确保输出不缓冲)
nohup python -u ${PROJECT_ROOT}/src/agents/offline/td3_bc_v2.py \
    --env_name ${ENV_NAME} \
    --dataset_path ${DATASET_PATH} \
    --seed ${SEED} \
    --alpha ${ALPHA} \
    --discount ${GAMMA} \
    --normalize_reward True \
    --log_dir ${LOG_DIR} \
    --checkpoint_dir ${CKPT_DIR} \
    --max_timesteps 1000000 \
    --batch_size 256 \
    --save_freq 100000 \
    --use_swanlab True \
    --swan_project "GeMS_Offline_RL_202512" \
    --swan_workspace "Cliff" \
    --swan_mode "cloud" \
    > ${LOG_FILE} 2>&1 &

# 保存PID
PID=$!
echo ${PID} > ${LOG_DIR}/td3bc_${ENV_NAME}_seed${SEED}.pid

echo "✅ Training started!"
echo "  PID: ${PID}"
echo "  Log file: ${LOG_FILE}"
echo ""
echo "📊 Monitor training:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "🛑 Stop training:"
echo "  kill ${PID}"
echo ""
echo "================================================================================"
