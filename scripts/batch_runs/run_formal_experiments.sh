#!/bin/bash
################################################################################
# 正式实验脚本 - BC & IQL 算法全面对比
#
# 实验设计：
#   - 算法: BC, IQL (2个)
#   - Ranker: gems, topk, kheadargmax (3个)
#   - 环境: mix_divpen, topdown_divpen (2个)
#   - 总计: 2 × 3 × 2 = 12个实验
#
# 参数配置：
#   - SEED: 58407201 (固定种子保证可复现)
#   - MAX_STEPS: 100000 (10万步训练)
#   - EVAL_FREQ: 1000 (每1000步评估一次，共100次评估)
#   - BATCH_SIZE: 256
#   - QUALITY: v2_b5 (高质量数据集)
#
# IQL特定参数：
#   - EXPECTILE: 0.7 (期望分位数)
#   - BETA: 1.0 (优势加权系数)
#
# 日志输出：
#   - 自动保存到: experiments/logs/offline/log_58407201/{BC|IQL}/
#   - 文件命名: {env_name}_{quality}_seed{seed}_{run_id}.log
#   - 无需手动重定向，代码内置日志管理
#
# GPU分配策略：
#   - BC实验: GPU 0-2 (3个ranker并行)
#   - IQL实验: GPU 3-5 (3个ranker并行)
#   - 每个环境的实验串行运行，避免资源冲突
#
# 运行方式：
#   bash scripts/batch_runs/run_formal_experiments.sh
#
# 作者: Claude Code
# 日期: 2026-02-13
################################################################################

set -e  # 遇到错误立即退出

# ============================================================================
# 全局配置
# ============================================================================

PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python
SEED=58407201
MAX_STEPS=100000
EVAL_FREQ=1000
BATCH_SIZE=256
QUALITY=v2_b5

# IQL特定参数
EXPECTILE=0.7
BETA=1.0

# GPU配置
BC_GPUS=(0 1 2)      # BC使用GPU 0-2
IQL_GPUS=(3 4 5)     # IQL使用GPU 3-5

# 环境列表
ENVS=(mix_divpen topdown_divpen)

# Ranker列表
RANKERS=(gems topk kheadargmax)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 辅助函数
# ============================================================================

print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# 实验启动函数
# ============================================================================

run_bc_experiments() {
    local env=$1
    local env_display=$(echo $env | sed 's/_/ /g')

    print_header "BC Experiments - ${env_display}"

    for i in "${!RANKERS[@]}"; do
        local ranker=${RANKERS[$i]}
        local gpu=${BC_GPUS[$i]}
        local exp_name="bc_${ranker}_${env}"

        print_info "[$(($i+1))/3] Starting: BC + ${ranker} + ${env} (GPU ${gpu})"

        nohup env CUDA_VISIBLE_DEVICES=$gpu $PYTHON src/agents/offline/bc.py \
            --experiment_name "$exp_name" \
            --env_name $env \
            --dataset_quality $QUALITY \
            --ranker $ranker \
            --max_timesteps $MAX_STEPS \
            --eval_freq $EVAL_FREQ \
            --batch_size $BATCH_SIZE \
            --seed $SEED \
            --device cuda:0 &

        sleep 3  # 避免同时启动导致资源冲突
    done

    print_info "All BC experiments for ${env} started!"
    echo ""
}

run_iql_experiments() {
    local env=$1
    local env_display=$(echo $env | sed 's/_/ /g')

    print_header "IQL Experiments - ${env_display}"

    for i in "${!RANKERS[@]}"; do
        local ranker=${RANKERS[$i]}
        local gpu=${IQL_GPUS[$i]}
        local exp_name="iql_${ranker}_${env}"

        print_info "[$(($i+1))/3] Starting: IQL + ${ranker} + ${env} (GPU ${gpu})"

        nohup env CUDA_VISIBLE_DEVICES=$gpu $PYTHON src/agents/offline/iql.py \
            --experiment_name "$exp_name" \
            --env_name $env \
            --dataset_quality $QUALITY \
            --ranker $ranker \
            --expectile $EXPECTILE \
            --beta $BETA \
            --max_timesteps $MAX_STEPS \
            --eval_freq $EVAL_FREQ \
            --batch_size $BATCH_SIZE \
            --seed $SEED \
            --device cuda:0 &

        sleep 3
    done

    print_info "All IQL experiments for ${env} started!"
    echo ""
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    print_header "正式实验启动 - BC & IQL 全面对比"

    echo -e "${YELLOW}实验配置:${NC}"
    echo "  算法: BC, IQL"
    echo "  Ranker: gems, topk, kheadargmax"
    echo "  环境: mix_divpen, topdown_divpen"
    echo "  训练步数: ${MAX_STEPS}"
    echo "  评估频率: ${EVAL_FREQ}"
    echo "  批次大小: ${BATCH_SIZE}"
    echo "  数据集质量: ${QUALITY}"
    echo "  种子: ${SEED}"
    echo "  IQL参数: expectile=${EXPECTILE}, beta=${BETA}"
    echo "  总实验数: 12"
    echo ""

    echo -e "${YELLOW}GPU分配:${NC}"
    echo "  BC: GPU ${BC_GPUS[@]}"
    echo "  IQL: GPU ${IQL_GPUS[@]}"
    echo ""

    echo -e "${YELLOW}日志位置:${NC}"
    echo "  BC: experiments/logs/offline/log_${SEED}/BC/"
    echo "  IQL: experiments/logs/offline/log_${SEED}/IQL/"
    echo ""

    read -p "确认启动所有实验? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "实验已取消"
        exit 0
    fi

    # ========================================================================
    # Phase 1: mix_divpen 环境实验
    # ========================================================================

    print_header "Phase 1: mix_divpen 环境实验 (6个实验)"

    # BC + mix_divpen (3个实验)
    run_bc_experiments "mix_divpen"

    # 等待BC实验启动完成
    sleep 5

    # IQL + mix_divpen (3个实验)
    run_iql_experiments "mix_divpen"

    # 等待mix_divpen实验稳定运行
    print_info "等待 mix_divpen 实验稳定运行..."
    sleep 10

    # ========================================================================
    # Phase 2: topdown_divpen 环境实验
    # ========================================================================

    print_header "Phase 2: topdown_divpen 环境实验 (6个实验)"

    # BC + topdown_divpen (3个实验)
    run_bc_experiments "topdown_divpen"

    # 等待BC实验启动完成
    sleep 5

    # IQL + topdown_divpen (3个实验)
    run_iql_experiments "topdown_divpen"

    # ========================================================================
    # 完成
    # ========================================================================

    print_header "所有实验已启动完成！"

    echo -e "${GREEN}实验状态:${NC}"
    echo "  ✅ BC + mix_divpen: 3个实验 (GPU ${BC_GPUS[@]})"
    echo "  ✅ IQL + mix_divpen: 3个实验 (GPU ${IQL_GPUS[@]})"
    echo "  ✅ BC + topdown_divpen: 3个实验 (GPU ${BC_GPUS[@]})"
    echo "  ✅ IQL + topdown_divpen: 3个实验 (GPU ${IQL_GPUS[@]})"
    echo ""

    echo -e "${YELLOW}查看实验状态:${NC}"
    echo "  # 查看所有后台任务"
    echo "  jobs"
    echo ""
    echo "  # 查看GPU使用情况"
    echo "  gpustat"
    echo ""
    echo "  # 查看BC日志"
    echo "  ls -lh experiments/logs/offline/log_${SEED}/BC/"
    echo "  tail -f experiments/logs/offline/log_${SEED}/BC/mix_divpen_v2_b5_seed${SEED}_*.log"
    echo ""
    echo "  # 查看IQL日志"
    echo "  ls -lh experiments/logs/offline/log_${SEED}/IQL/"
    echo "  tail -f experiments/logs/offline/log_${SEED}/IQL/mix_divpen_v2_b5_seed${SEED}_*.log"
    echo ""

    echo -e "${YELLOW}SwanLab监控:${NC}"
    echo "  项目地址: https://swanlab.cn/@Cliff/Offline_Slate_RL_202602"
    echo ""

    echo -e "${GREEN}预计完成时间: 2-4小时${NC}"
    echo ""
}

# 运行主程序
main
