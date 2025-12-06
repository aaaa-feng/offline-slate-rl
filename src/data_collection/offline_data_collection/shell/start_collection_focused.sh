#!/bin/bash
# Focused环境数据收集启动脚本
# 使用方法: bash start_collection_focused.sh

# 获取当前日期时间
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 项目根目录 - 使用新的项目路径
PROJECT_ROOT="/data/liyuefeng/offline-slate-rl"
COLLECTION_DIR="${PROJECT_ROOT}/src/data_collection/offline_data_collection"
LOG_DIR="${PROJECT_ROOT}/experiments/logs/offline_data_collection"

# 确保log目录存在
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "开始启动Focused环境数据收集任务"
echo "时间戳: ${TIMESTAMP}"
echo "项目根目录: ${PROJECT_ROOT}"
echo "=========================================="
echo ""

# 启动三个focused环境的数据收集
echo "启动 focused_topdown 数据收集 (GPU 5)..."
cd ${COLLECTION_DIR}
nohup bash -c "source /data/liyuefeng/miniconda3/etc/profile.d/conda.sh && conda activate gems && python scripts/collect_data.py --env_name focused_topdown --episodes 10000 --output_dir ${PROJECT_ROOT}/datasets/offline_datasets --gpu 5" > ${LOG_DIR}/collect_focused_topdown_${TIMESTAMP}.log 2>&1 &
PID1=$!
echo "  PID: ${PID1}"
echo "  日志: ${LOG_DIR}/collect_focused_topdown_${TIMESTAMP}.log"
echo ""

sleep 2

echo "启动 focused_mix 数据收集 (GPU 6)..."
nohup bash -c "source /data/liyuefeng/miniconda3/etc/profile.d/conda.sh && conda activate gems && python scripts/collect_data.py --env_name focused_mix --episodes 10000 --output_dir ${PROJECT_ROOT}/datasets/offline_datasets --gpu 6" > ${LOG_DIR}/collect_focused_mix_${TIMESTAMP}.log 2>&1 &
PID2=$!
echo "  PID: ${PID2}"
echo "  日志: ${LOG_DIR}/collect_focused_mix_${TIMESTAMP}.log"
echo ""

sleep 2

echo "启动 focused_divpen 数据收集 (GPU 7)..."
nohup bash -c "source /data/liyuefeng/miniconda3/etc/profile.d/conda.sh && conda activate gems && python scripts/collect_data.py --env_name focused_divpen --episodes 10000 --output_dir ${PROJECT_ROOT}/datasets/offline_datasets --gpu 7" > ${LOG_DIR}/collect_focused_divpen_${TIMESTAMP}.log 2>&1 &
PID3=$!
echo "  PID: ${PID3}"
echo "  日志: ${LOG_DIR}/collect_focused_divpen_${TIMESTAMP}.log"
echo ""

echo "=========================================="
echo "所有Focused环境任务已启动"
echo "=========================================="
echo ""
echo "进程列表:"
echo "  focused_topdown: PID ${PID1}"
echo "  focused_mix:     PID ${PID2}"
echo "  focused_divpen:  PID ${PID3}"
echo ""
echo "查看日志:"
echo "  tail -f ${LOG_DIR}/collect_focused_topdown_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/collect_focused_mix_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/collect_focused_divpen_${TIMESTAMP}.log"
echo ""
echo "监控进程:"
echo "  ps aux | grep collect_data.py"
echo ""
