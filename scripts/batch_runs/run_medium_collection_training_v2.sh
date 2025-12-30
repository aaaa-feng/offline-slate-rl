#!/bin/bash

# =================================================================
# Medium 数据集模型训练脚本 V2 (50k steps)
# =================================================================
# 功能：
# 1. 在 9 个实验配置上并行训练 SAC+GeMS
#    - 3 个 diffuse 环境 (beta=1.0, lambda_click=0.5)
#    - 6 个 focused 环境 (两套参数各3个环境)
#      * 参数组1: beta=1.0, lambda_click=0.5
#      * 参数组2: beta=0.5, lambda_click=0.2
# 2. 强制在 50,000 步保存模型 (用于收集 Medium 数据)
# 3. 自动分配任务到 GPU 1, 2, 3
# =================================================================

# 0. 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gems

# 1. 基础配置
# GPU分配策略：9个实验分配到7个GPU (0-6)
# GPU 0: 3个任务, GPU 1-6: 各1个任务, GPU 7: 不使用
GPU_IDS=(0 0 0 1 2 3 4 5 6)  # 9个任务对应的GPU
SAVE_STEP=50000
MAX_STEPS=100000
SEED=58407201
EXP_PURPOSE="medium_collection"

# 生成时间戳和日志目录
TIMESTAMP=$(date +%Y%m%d)
LOG_BASE_DIR="/data/liyuefeng/offline-slate-rl/experiments/logs/log_${SEED}/SAC_GeMS/${EXP_PURPOSE}_${TIMESTAMP}"

# 确保日志目录存在
mkdir -p ${LOG_BASE_DIR}

# 2. 定义实验配置列表
# 格式: "环境名 lambda_KL lambda_click"
EXPERIMENTS=(
    # Diffuse 环境 (3个) - 只用一套参数
    "diffuse_topdown 1.0 0.5"
    "diffuse_mix 1.0 0.5"
    "diffuse_divpen 1.0 0.5"

    # Focused 环境 - 参数组1 (beta=1.0, lambda_click=0.5)
    "focused_topdown 1.0 0.5"
    "focused_mix 1.0 0.5"
    "focused_divpen 1.0 0.5"

    # Focused 环境 - 参数组2 (beta=0.5, lambda_click=0.2)
    "focused_topdown 0.5 0.2"
    "focused_mix 0.5 0.2"
    "focused_divpen 0.5 0.2"
)

echo "========================================================================"
echo "=== 开始训练 Medium (50k step) 模型 - V2 (9个实验配置) ==="
echo "========================================================================"
echo "实验配置:"
echo "  - Diffuse 环境: 3个 (beta=1.0, lambda_click=0.5)"
echo "  - Focused 环境: 6个 (两套参数各3个)"
echo "    * 参数组1: beta=1.0, lambda_click=0.5"
echo "    * 参数组2: beta=0.5, lambda_click=0.2"
echo "========================================================================"
echo "模型将保存到: checkpoints/online_rl/{env_name}/"
echo "日志将保存到: ${LOG_BASE_DIR}/"
echo "========================================================================"
echo ""

# 3. 循环启动任务
for i in "${!EXPERIMENTS[@]}"; do
    # 解析实验配置
    read -r ENV LAMBDA_KL LAMBDA_CLICK <<< "${EXPERIMENTS[$i]}"

    # --- 自动分配 GPU (直接使用数组索引) ---
    GPU_ID=${GPU_IDS[$i]}

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
    LOG_FILE="${LOG_BASE_DIR}/${ENV}_KL${LAMBDA_KL}_click${LAMBDA_CLICK}_${TIMESTAMP}.log"

    echo "----------------------------------------------------------------"
    echo "启动任务 [$((i+1))/9]: ${ENV}"
    echo "  - GPU: ${GPU_ID}"
    echo "  - Lambda KL (beta): ${LAMBDA_KL}"
    echo "  - Lambda Click: ${LAMBDA_CLICK}"
    echo "  - Click Model: ${CLICK_MODEL}"
    echo "  - Diversity Penalty: ${DIV_PENALTY}"
    echo "  - Env Embeds: ${ENV_EMBEDDS}"
    echo "  - Log: ${LOG_FILE}"
    echo "  - Save: checkpoints/online_rl/${ENV}/"
    echo "----------------------------------------------------------------"

    # --- 执行训练命令 ---
    # 使用 CUDA_VISIBLE_DEVICES 隔离显卡
    # 使用 nohup 后台运行
    # 注意：需要在项目根目录下运行，并激活conda环境

    (source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate gems && \
    cd /data/liyuefeng/offline-slate-rl && \
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
        --lambda_KL=${LAMBDA_KL} \
        --lambda_click=${LAMBDA_CLICK} \
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
        --swan_tags "medium_collection" "${ENV}" "seed_${SEED}" "50k_steps" "KL${LAMBDA_KL}" "click${LAMBDA_CLICK}" \
        --swan_description="SAC+GeMS Medium Data Collection - ${ENV} - KL${LAMBDA_KL} click${LAMBDA_CLICK} - seed ${SEED} - 50k steps" \
        --run_name="SAC_GeMS_Medium_${ENV}_KL${LAMBDA_KL}_click${LAMBDA_CLICK}_seed${SEED}" \
        --progress_bar=False \
        > "${LOG_FILE}" 2>&1 &)

    # 稍微暂停一下，避免同时启动冲击 CPU
    sleep 2
done

echo ""
echo "========================================================================"
echo "🎉 所有 9 个任务已启动!"
echo "========================================================================"
echo "📁 日志目录: ${LOG_BASE_DIR}/"
echo "💾 模型保存: checkpoints/online_rl/{env_name}/"
echo ""
echo "实验配置总结:"
echo "  1-3:   Diffuse 环境 (KL1.0, click0.5)"
echo "  4-6:   Focused 环境 - 参数组1 (KL1.0, click0.5)"
echo "  7-9:   Focused 环境 - 参数组2 (KL0.5, click0.2)"
echo "========================================================================"
echo ""
echo "可以使用以下命令查看进度:"
echo "  - tail -f ${LOG_BASE_DIR}/*.log          # 查看训练日志"
echo "  - ls ${LOG_BASE_DIR}/                    # 列出所有日志文件"
echo "  - grep 'Training Step' ${LOG_BASE_DIR}/*.log | tail -20  # 查看最新训练进度"
echo "  - grep 'Loading Pretrained GeMS' ${LOG_BASE_DIR}/*.log  # 检查GeMS加载情况"
echo ""
