#!/bin/bash
# Batch Offline GeMS Training Script
# 针对6个数据集（3个环境 × 2个质量级别）进行训练
# 分别挂在GPU 0-5号卡上
#
# 参数配置参考: config/online/gems/pretrain_gems.yml
# - lambda_click: 0.5
# - lambda_KL: 1.0
# - lambda_prior: 0.0
# - latent_dim: 32
# - batch_size: 256
# - ranker_lr: 0.001
# - max_epochs: 50
# - seed: 58407201

# 设置工作目录
cd /data/liyuefeng/offline-slate-rl

# 获取今天的日期
DATE=$(date +%Y%m%d)

# 固定参数
SEED=58407201
LATENT_DIM=32
LAMBDA_KL=1.0
LAMBDA_CLICK=0.5
LAMBDA_PRIOR=0.0
RANKER_LR=0.001
MAX_EPOCHS=50
BATCH_SIZE=256
NUM_WORKERS=0

# 创建主日志目录
LOG_BASE_DIR="experiments/logs/offline/log_${SEED}/pretrain_gems"
mkdir -p ${LOG_BASE_DIR}

echo "=========================================="
echo "Batch Offline GeMS Training"
echo "=========================================="
echo "Date: ${DATE}"
echo "Seed: ${SEED}"
echo "Log directory: ${LOG_BASE_DIR}"
echo "=========================================="
echo ""

# 定义6个训练任务
# 格式: GPU_ID ENV_NAME QUALITY EMBEDDING_PATH
TASKS=(
    "0 diffuse_topdown expert data/embeddings/item_embeddings_diffuse.pt"
    "1 diffuse_topdown medium data/embeddings/item_embeddings_diffuse.pt"
    "2 diffuse_mix expert data/embeddings/item_embeddings_diffuse.pt"
    "3 diffuse_mix medium data/embeddings/item_embeddings_diffuse.pt"
    "4 diffuse_divpen expert data/embeddings/item_embeddings_diffuse.pt"
    "5 diffuse_divpen medium data/embeddings/item_embeddings_diffuse.pt"
)

# 启动所有训练任务
for task in "${TASKS[@]}"; do
    # 解析任务参数
    read -r GPU_ID ENV_NAME QUALITY EMBEDDING_PATH <<< "$task"

    # 日志文件路径
    LOG_FILE="${LOG_BASE_DIR}/train_${ENV_NAME}_${QUALITY}_${DATE}.log"

    echo "Starting training on GPU ${GPU_ID}:"
    echo "  Environment: ${ENV_NAME}"
    echo "  Quality: ${QUALITY}"
    echo "  Log file: ${LOG_FILE}"

    # 使用CUDA_VISIBLE_DEVICES指定GPU，启动训练
    CUDA_VISIBLE_DEVICES=${GPU_ID} nohup /data/liyuefeng/miniconda3/envs/gems/bin/python scripts/train_gems_offline.py \
        --env_name ${ENV_NAME} \
        --quality ${QUALITY} \
        --embedding_path ${EMBEDDING_PATH} \
        --latent_dim ${LATENT_DIM} \
        --lambda_KL ${LAMBDA_KL} \
        --lambda_click ${LAMBDA_CLICK} \
        --lambda_prior ${LAMBDA_PRIOR} \
        --ranker_lr ${RANKER_LR} \
        --max_epochs ${MAX_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS} \
        --seed ${SEED} \
        --progress_bar \
        --swan_mode disabled \
        > ${LOG_FILE} 2>&1 &

    # 获取进程ID
    PID=$!
    echo "  PID: ${PID}"
    echo ""

    # 短暂延迟，避免同时启动导致资源竞争
    sleep 5
done

echo "=========================================="
echo "All training jobs started!"
echo "=========================================="
echo ""
echo "To monitor all logs:"
echo "  tail -f ${LOG_BASE_DIR}/train_*_${DATE}.log"
echo ""
echo "To check running processes:"
echo "  ps aux | grep train_gems_offline"
echo ""
echo "To monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Log files:"
for task in "${TASKS[@]}"; do
    read -r GPU_ID ENV_NAME QUALITY EMBEDDING_PATH <<< "$task"
    echo "  GPU ${GPU_ID}: ${LOG_BASE_DIR}/train_${ENV_NAME}_${QUALITY}_${DATE}.log"
done
echo ""
