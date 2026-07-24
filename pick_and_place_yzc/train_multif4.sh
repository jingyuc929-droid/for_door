#!/bin/bash

# 设置tmux会话名称
SESSION_NAME="f4kp90kd3d5"

# 定义要执行的命令序列（最后一个是无限循环命令）
COMMANDS=(
    "echo 'Running my Python script...'"
    "agenton"
    "source /home/user_wh/anaconda3/bin/activate loco"
    "export LD_LIBRARY_PATH=/home/user_wh/anaconda3/envs/loco/lib"
    "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    "export WANDB_USERNAME=2273425240-gll"
    "export JAX_LOCAL_RANK=0"
    "python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:26292 \
    scripts/locomotion/train.py \
    --task Locomotion-GRQ20-V2D4-X5-Smooth-Terrain-VAE \
    --run_name=$SESSION_NAME \
    --device cuda:0 \
    --headless --distributed"  # 假设这是无限循环命令
)

# COMMANDS=(
#     "echo 'Starting training script...'",
#     "echo 'Activating conda environment...'",
#     "agenton"
# )

# 检查会话是否存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "会话 $SESSION_NAME 已存在，进入会话并重启命令..."
    
    # 发送Ctrl+C终止当前可能正在运行的无限循环命令
    tmux send-keys -t "$SESSION_NAME" C-c
    
    # 等待片刻让命令彻底终止（根据实际情况调整时间）
    sleep 5
    
    # 发送打包的命令到会话
    for cmd in "${COMMANDS[@]}"; do
        tmux send-keys -t "$SESSION_NAME" "$cmd" C-m
        
        # 对于最后一个命令（无限循环）不等待，其他命令短暂等待
        if [ "$cmd" != "${COMMANDS[-1]}" ]; then
            sleep 0.5
        fi
    done
    
    # 进入会话
    tmux attach -t "$SESSION_NAME"
    
else
    echo "创建新会话 $SESSION_NAME 并执行命令..."
    # 创建新会话并在后台运行
    tmux new-session -d -t "$SESSION_NAME"
    
    # 发送打包的命令到新会话
    for cmd in "${COMMANDS[@]}"; do
        tmux send-keys -t "$SESSION_NAME" "$cmd" C-m
        
        # 对于最后一个命令（无限循环）不等待，其他命令短暂等待
        if [ "$cmd" != "${COMMANDS[-1]}" ]; then
            sleep 0.5
        fi
    done
    
    # 进入创建的会话
    tmux attach -t "$SESSION_NAME"
fi