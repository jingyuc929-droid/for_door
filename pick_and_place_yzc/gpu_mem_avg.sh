#!/bin/bash

# 配置参数：采集间隔（秒）、总运行时长（秒）
INTERVAL=2
DURATION=600  # 示例：运行60秒，可自行修改（需是INTERVAL的整数倍）
TOTAL_STEPS=$((DURATION / INTERVAL))

# 初始化显存累加数组（适配多卡，自动识别GPU数量）
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
declare -a MEM_USAGE_SUM
for ((i=0; i<GPU_COUNT; i++)); do
    MEM_USAGE_SUM[$i]=0
done

echo "开始采集显存数据（间隔${INTERVAL}秒，总时长${DURATION}秒）..."
echo "========================================"

# 循环采集显存数据
for ((step=1; step<=TOTAL_STEPS; step++)); do
    # 提取每块GPU的已用显存（单位：MiB）
    MEM_USAGE_LIST=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    
    # 逐卡累加显存值
    GPU_INDEX=0
    while IFS= read -r mem_used; do
        MEM_USAGE_SUM[$GPU_INDEX]=$((MEM_USAGE_SUM[$GPU_INDEX] + mem_used))
        GPU_INDEX=$((GPU_INDEX + 1))
    done <<< "$MEM_USAGE_LIST"

    # 打印实时进度（可选）
    echo "第${step}/${TOTAL_STEPS}次采集完成"
    sleep $INTERVAL
done

# 计算并输出各GPU的显存均值
echo "========================================"
echo "采集完成！各GPU显存使用均值（单位：MiB）："
for ((i=0; i<GPU_COUNT; i++)); do
    AVG_MEM=$((MEM_USAGE_SUM[$i] / TOTAL_STEPS))
    echo "GPU ${i}: ${AVG_MEM} MiB"
done