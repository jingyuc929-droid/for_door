#!/usr/bin/env bash
# ============================================================
# 从远程训练服务器增量拉取全部 locomotion 训练产物到本机，供 play / 测试使用。
#
# 用法：
#   1. 填写下面的服务器连接和 REMOTE_PROJECT_DIR（默认与 sync_remote.sh 一致）
#   2. 空跑确认：DRY_RUN=1 ./pull_remote_weights.sh
#   3. 正式下载：  ./pull_remote_weights.sh
#
# 默认同步 logs/locomotion 下所有实验和 run；rsync 会跳过未变化的文件，
# 因此可反复执行，也不需要为新的实验名称改脚本。
# 传整个 run 而非单个 model_*.pt：其中 params/env.pkl、params/agent.pkl
# 保存着训练时的配置，play.py 会优先读取它们，能避免配置变动后的维度不匹配。
# ============================================================
set -euo pipefail

# ===== 服务器连接（与 sync_remote.sh 保持一致）================
REMOTE_USER="szy"
REMOTE_HOST="192.168.161.107"
REMOTE_PROJECT_DIR="~/rl_sim_env_dev_yzc"
# ================================================================

LOCAL_PROJECT_DIR="/home/sun/rl_sim_env_dev_yzc"

if [[ $# -ne 0 ]]; then
  echo "用法: $0" >&2
  exit 2
fi

REMOTE_LOG_ROOT="$REMOTE_PROJECT_DIR/logs/locomotion"
LOCAL_LOG_ROOT="$LOCAL_PROJECT_DIR/logs/locomotion"

DRY_RUN_FLAGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_FLAGS=(-n -v)
fi

echo "==> $REMOTE_USER@$REMOTE_HOST:$REMOTE_LOG_ROOT/"
echo "    → $LOCAL_LOG_ROOT/"
[[ ${#DRY_RUN_FLAGS[@]} -gt 0 ]] && echo "==> 空跑模式（DRY_RUN=1），只列出将下载的文件"

mkdir -p "$LOCAL_LOG_ROOT"
rsync -azh --partial --info=progress2 \
  "${DRY_RUN_FLAGS[@]}" \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_LOG_ROOT/" \
  "$LOCAL_LOG_ROOT/"

echo "==> 权重及训练配置已拉取完成"
