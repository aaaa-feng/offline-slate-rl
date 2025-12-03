#!/bin/bash
# 数据收集启动脚本
# 使用方法: bash start_collection.sh

# 获取当前日期时间
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 项目根目录
PROJECT_ROOT="/data/liyuefeng/gems/gems_official/official_code"
LOG_DIR="${PROJECT_ROOT}/offline_data_collection/logs"

# 确保log目录存在
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "开始启动数据收集任务"
echo "时间戳: ${TIMESTAMP}"
echo "=========================================="
echo ""

# 启动三个环境的数据收集 (全部使用GPU 5)
echo "启动 diffuse_topdown 数据收集 (GPU 5)..."
cd ${PROJECT_ROOT}
nohup bash -c "source /data/liyuefeng/miniconda3/etc/profile.d/conda.sh && conda activate gems && python offline_data_collection/collect_data.py --env_name diffuse_topdown --episodes 10000 --output_dir ./offline_datasets --gpu 1" > ${LOG_DIR}/collect_diffuse_topdown_${TIMESTAMP}.log 2>&1 &
PID1=$!
echo "  PID: ${PID1}"
echo "  日志: ${LOG_DIR}/collect_diffuse_topdown_${TIMESTAMP}.log"
echo ""

sleep 2

echo "启动 diffuse_mix 数据收集 (GPU 5)..."
nohup bash -c "source /data/liyuefeng/miniconda3/etc/profile.d/conda.sh && conda activate gems && python offline_data_collection/collect_data.py --env_name diffuse_mix --episodes 10000 --output_dir ./offline_datasets --gpu 1" > ${LOG_DIR}/collect_diffuse_mix_${TIMESTAMP}.log 2>&1 &
PID2=$!
echo "  PID: ${PID2}"
echo "  日志: ${LOG_DIR}/collect_diffuse_mix_${TIMESTAMP}.log"
echo ""

sleep 2

echo "启动 diffuse_divpen 数据收集 (GPU 5)..."
nohup bash -c "source /data/liyuefeng/miniconda3/etc/profile.d/conda.sh && conda activate gems && python offline_data_collection/collect_data.py --env_name diffuse_divpen --episodes 10000 --output_dir ./offline_datasets --gpu 1" > ${LOG_DIR}/collect_diffuse_divpen_${TIMESTAMP}.log 2>&1 &
PID3=$!
echo "  PID: ${PID3}"
echo "  日志: ${LOG_DIR}/collect_diffuse_divpen_${TIMESTAMP}.log"
echo ""

echo "=========================================="
echo "所有任务已启动"
echo "=========================================="
echo ""
echo "进程ID:"
echo "  diffuse_topdown: ${PID1}"
echo "  diffuse_mix: ${PID2}"
echo "  diffuse_divpen: ${PID3}"
echo ""
echo "查看日志:"
echo "  tail -f ${LOG_DIR}/collect_diffuse_topdown_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/collect_diffuse_mix_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/collect_diffuse_divpen_${TIMESTAMP}.log"
echo ""
echo "查看进程状态:"
echo "  ps aux | grep collect_data.py"
echo ""
echo "停止所有任务:"
echo "  kill ${PID1} ${PID2} ${PID3}"
echo ""
