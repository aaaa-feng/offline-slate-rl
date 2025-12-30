#!/bin/bash
################################################################################
# V2数据全量收集脚本
# 功能：收集包含raw_obs和Oracle信息的完整数据集
# 输出：data/datasets/offline_v2/
#
# 新增字段：
# - raw_observations: 原始环境观察（字典格式）
# - raw_next_observations: 下一步原始观察
# - user_states: 用户状态（从raw_obs提取）
# - user_bored: 用户厌倦标志
# - item_relevances: Oracle信息（物品相关性，ground truth）
#
# GPU分配策略：
# - 6个任务分配到GPU 0-5（每卡一个任务，避免显存冲突）
# - 每个任务预计使用约3GB显存
#
# 使用方法：
#   bash scripts/batch_runs/collect_all_v2.sh
################################################################################

set -e  # 遇到错误立即退出

# ============================================================================
# 配置区域
# ============================================================================
PYTHON_BIN="/data/liyuefeng/miniconda3/envs/gems/bin/python"
PROJECT_ROOT="/data/liyuefeng/offline-slate-rl"
SCRIPT_PATH="${PROJECT_ROOT}/src/data_collection/offline_data_collection/collect_data.py"
OUTPUT_DIR="${PROJECT_ROOT}/data/datasets/offline_v2"
LOG_DIR="${PROJECT_ROOT}/experiments/logs/offline_data_collection/20251230"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# 时间戳（用于日志文件名）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "  V2数据全量收集 - 开始"
echo "================================================================================"
echo "时间: $(date)"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志目录: ${LOG_DIR}"
echo "Python: ${PYTHON_BIN}"
echo ""
echo "收集配置:"
echo "  - 环境: diffuse_topdown, diffuse_mix, diffuse_divpen (3个)"
echo "  - 质量: expert, medium (2个)"
echo "  - Episodes: 10,000 per quality"
echo "  - 总计: 6个数据集"
echo "  - 新增: raw_obs + Oracle信息 (--save_raw_obs)"
echo ""
echo "GPU分配:"
echo "  - GPU 0: diffuse_topdown expert"
echo "  - GPU 1: diffuse_topdown medium"
echo "  - GPU 2: diffuse_mix expert"
echo "  - GPU 3: diffuse_mix medium"
echo "  - GPU 4: diffuse_divpen expert"
echo "  - GPU 5: diffuse_divpen medium"
echo "================================================================================"
echo ""

# ============================================================================
# 任务定义
# ============================================================================
# 格式: "GPU_ID ENV_NAME QUALITY"
TASKS=(
    "0 diffuse_topdown expert"
    "1 diffuse_topdown medium"
    "2 diffuse_mix expert"
    "3 diffuse_mix medium"
    "4 diffuse_divpen expert"
    "5 diffuse_divpen medium"
)

# ============================================================================
# 启动所有任务
# ============================================================================
echo "开始启动所有收集任务..."
echo ""

for task in "${TASKS[@]}"; do
    # 解析任务参数
    read -r gpu_id env_name quality <<< "$task"

    # 生成日志文件名（格式: collect_{env}_{quality}_20251230_HHMMSS.log）
    log_file="${LOG_DIR}/collect_${env_name}_${quality}_${TIMESTAMP}.log"

    echo "--------------------------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] 启动任务: ${env_name} - ${quality} (GPU ${gpu_id})"
    echo "  日志文件: ${log_file}"
    echo "--------------------------------------------------------------------------------"

    # 使用nohup后台运行，指定GPU
    CUDA_VISIBLE_DEVICES=${gpu_id} nohup ${PYTHON_BIN} ${SCRIPT_PATH} \
        --env_name "${env_name}" \
        --quality "${quality}" \
        --episodes 10000 \
        --output_dir "${OUTPUT_DIR}" \
        --save_raw_obs \
        > "${log_file}" 2>&1 &

    # 记录进程ID
    pid=$!
    echo "  进程ID: ${pid}"
    echo "  GPU: ${gpu_id}"
    echo ""

    # 短暂延迟，避免同时启动导致资源竞争
    sleep 2
done

echo "================================================================================"
echo "  所有任务已启动"
echo "================================================================================"
echo "时间: $(date)"
echo ""
echo "后台进程列表:"
jobs -l
echo ""
echo "监控命令:"
echo "  - 查看所有日志: tail -f ${LOG_DIR}/*.log"
echo "  - 查看特定任务: tail -f ${LOG_DIR}/collect_{env}_{quality}_${TIMESTAMP}.log"
echo "  - 查看GPU使用: watch -n 1 nvidia-smi"
echo "  - 查看进程: ps aux | grep collect_data.py"
echo ""
echo "注意事项:"
echo "  - 所有任务在后台运行，关闭终端不会中断"
echo "  - 预计每个任务运行时间: 约12-15小时（10k episodes）"
echo "  - 总预计完成时间: 约12-15小时（并行执行）"
echo "================================================================================"
