#!/bin/bash

# =================================================================
# Medium 数据集模型训练脚本 (50k steps)
# =================================================================
# 功能：
# 1. 在 6 个环境上并行训练 SAC+GeMS
# 2. 强制在 50,000 步保存模型 (用于收集 Medium 数据)
# 3. 自动分配任务到 GPU 5, 6, 7
# =================================================================

# 0. 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gems

# 1. 基础配置
GPU_IDS=(1 2 3)
SAVE_STEP=50000
MAX_STEPS=100000
SEED=58407201
EXP_PURPOSE="medium_collection"

# 生成时间戳和日志目录
TIMESTAMP=$(date +%Y%m%d)
LOG_BASE_DIR="/data/liyuefeng/offline-slate-rl/experiments/logs/log_${SEED}/SAC_GeMS/${EXP_PURPOSE}_${TIMESTAMP}"

# 确保日志目录存在
mkdir -p ${LOG_BASE_DIR}

# 2. 定义环境列表
ENVS=(
    "diffuse_topdown"
    "diffuse_mix"
    "diffuse_divpen"
    "focused_topdown"
    "focused_mix"
    "focused_divpen"
)

echo "=== 开始训练 Medium (50k step) 模型 ==="
echo "=== 模型将保存到: checkpoints/online_rl/{env_name}/ ==="
echo "=== 日志将保存到: ${LOG_BASE_DIR}/ ==="

# 3. 循环启动任务
for i in "${!ENVS[@]}"; do
    ENV=${ENVS[$i]}

    # --- 自动分配 GPU (轮询 5, 6, 7) ---
    GPU_IDX=$((i % 3))
    GPU_ID=${GPU_IDS[$GPU_IDX]}

    # --- 根据环境名判断参数 ---
    
    # 1. Click Model & Diversity Penalty
    if [[ "$ENV" == *"topdown"* ]]; then
        CLICK_MODEL="tdPBM"
        DIV_PENALTY=1.0
    elif [[ "$ENV" == *"mix"* ]]; then
        CLICK_MODEL="mixPBM"
        DIV_PENALTY=1.0
    elif [[ "$ENV" == *"divpen"* ]]; then
        CLICK_MODEL="mixPBM"
        DIV_PENALTY=3.0
    fi
    
    # 2. Environment Embeddings (Diffuse vs Focused)
    if [[ "$ENV" == *"diffuse"* ]]; then
        ENV_EMBEDDS="item_embeddings_diffuse.pt"
    else
        ENV_EMBEDDS="item_embeddings_focused.pt"
    fi

    # --- 生成日志文件名 ---
    # 格式: {env}_KL{lambda_KL}_click{lambda_click}_{timestamp}.log
    LOG_FILE="${LOG_BASE_DIR}/${ENV}_KL1.0_click0.5_${TIMESTAMP}.log"

    echo "----------------------------------------------------------------"
    echo "启动任务: ${ENV}"
    echo "  - GPU: ${GPU_ID}"
    echo "  - Click Model: ${CLICK_MODEL}"
    echo "  - Env Embeds: ${ENV_EMBEDDS}"
    echo "  - Log: ${LOG_FILE}"
    echo "  - Save: checkpoints/online_rl/${ENV}/"
    echo "----------------------------------------------------------------"

    # --- 执行训练命令 ---
    # 使用 CUDA_VISIBLE_DEVICES 隔离显卡
    # 使用 nohup 后台运行

    CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u scripts/train_online_rl.py \
        --agent=SAC \
        --belief=GRU \
        --ranker=GeMS \
        --item_embedds=scratch \
        --env_name=topics \
        --device=cuda \
        --seed=${SEED} \
        --ranker_seed=${SEED} \
        --max_steps=${MAX_STEPS} \
        --save_every_n_steps=${SAVE_STEP} \
        --check_val_every_n_epoch=1000 \
        --val_step_length=200 \
        --test_size=500 \
        --name="SAC+GeMS_Medium" \
        --exp_purpose=${EXP_PURPOSE} \
        --latent_dim=32 \
        --lambda_KL=1.0 \
        --lambda_click=0.5 \
        --lambda_prior=0.0 \
        --ranker_embedds=scratch \
        --ranker_sample=False \
        --ranker_dataset=${ENV} \
        --click_model=${CLICK_MODEL} \
        --env_embedds=${ENV_EMBEDDS} \
        --diversity_penalty=${DIV_PENALTY} \
        --capacity=10000 \
        --batch_size=20 \
        --random_steps=2000 \
        --q_lr=0.001 \
        --hidden_layers_qnet 256 \
        --target_update_frequency=1 \
        --tau=0.002 \
        --pi_lr=0.003 \
        --hidden_layers_pinet 256 \
        --gamma=0.8 \
        --auto_entropy=True \
        --alpha=0.2 \
        --belief_state_dim=20 \
        --item_embedd_dim=20 \
        --num_items=1000 \
        --boredom_threshold=5 \
        --recent_items_maxlen=10 \
        --boredom_moving_window=5 \
        --env_omega=0.9 \
        --short_term_boost=1.0 \
        --episode_length=100 \
        --env_offset=0.28 \
        --env_slope=100 \
        --diversity_threshold=4 \
        --topic_size=2 \
        --num_topics=10 \
        --beliefs actor critic \
        --swan_project="GeMS_RL_Training_202512" \
        --swan_mode=cloud \
        --swan_workspace="Cliff" \
        --swan_tags "medium_collection" "${ENV}" "seed_${SEED}" "50k_steps" \
        --swan_description="SAC+GeMS Medium Data Collection - ${ENV} - seed ${SEED} - 50k steps" \
        --run_name="SAC_GeMS_Medium_${ENV}_KL1.0_click0.5_seed${SEED}" \
        --progress_bar=False \
        > "${LOG_FILE}" 2>&1 &
        
    # 稍微暂停一下，避免同时启动冲击 CPU
    sleep 2
done

echo ""
echo "🎉 所有任务已挂起!"
echo "📁 日志目录: ${LOG_BASE_DIR}/"
echo "💾 模型保存: checkpoints/online_rl/{env_name}/"
echo ""
echo "可以使用以下命令查看进度:"
echo "  - tail -f ${LOG_BASE_DIR}/*.log          # 查看训练日志"
echo "  - ls ${LOG_BASE_DIR}/                    # 列出所有日志文件"
echo "  - grep 'Training Step' ${LOG_BASE_DIR}/*.log | tail -20  # 查看最新训练进度"
