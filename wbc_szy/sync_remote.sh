#!/usr/bin/env bash
# ============================================================
# 把本地 rl_sim_env_dev_yzc 增量同步到远程服务器
#
# 自动排除：日志、所有 .git 历史、代码未引用的机器人模型
# 保留：   全部代码/配置、datasets 全部工作文件、实际在用的机器人
#
# 扫描结论（数据来源：代码中的 usd_path / asset_path）：
#   - 在用的机器人仅 4 个，见下方 KEEP_ROBOTS
#   - datasets 实际数据仅 50M，全部保留
#   - 排除日志(10.2G) + 所有 .git(5.6G) + 其余机器人(925M) 后，约传 330M
#
# 用法：
#   1. 先填下面 [服务器连接] 三个变量
#   2. 空跑看清单（不实际传输）：  DRY_RUN=1 ./sync_remote.sh
#   3. 确认无误后正式同步：        ./sync_remote.sh
#
# 提示：
#   - rsync 是增量的，第二次起只传改过的文件，几乎瞬间完成
#   - 想让远程和本地完全一致（删掉远程多余文件）取消最后一行 --delete 的注释，谨慎！
#   - 以后换了机器人型号，在 KEEP_ROBOTS 里加名字即可
# ============================================================
set -euo pipefail

# ===== 服务器连接（来自 ~/.ssh/config：Host 192.168.161.107 / Port 2222 / User szy）=====
REMOTE_USER="szy"                  # 远程用户名
REMOTE_HOST="192.168.161.107"      # 用 ssh config 里的 Host 别名，端口 2222 会自动生效
REMOTE_DIR="~/rl_sim_env_dev_yzc"  # 远程目标目录；想放别的盘/路径就改这里
# ======================================================================================

# 源目录末尾的 / 必须保留：表示「同步目录内容」而非「把目录本身传过去」
LOCAL_SRC="/home/sun/rl_sim_env_dev_yzc/"
ROBOTS_BASE="source/rl_sim_env/data/assets/robots"

# 实际在用的机器人（扫描代码 usd_path/asset_path 得出；换型号就改这里）
KEEP_ROBOTS=(
  grq20_v2d4_piperL
  grq20_v2d4_piperL_front_mount
  grq20_v2d4_piperL_front_mount_gripper
  grq20_v2d4_x5
  grq20_v2d5_piperL_fixedgrasper
)

# ---- 组装 rsync 过滤规则（顺序敏感：先 include 后 exclude）----
FILTERS=(
  # 日志 / 实验产物
  --exclude='log/'
  --exclude='logs/'
  --exclude='swanlog/'           # SwanLab 实验日志
  --exclude='outputs/'
  # 所有 git 历史（含 assets/datasets 里嵌套的，省 5.6G）
  --exclude='.git/'
  # 编译 / 缓存
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.DS_Store'
)

# 白名单：只保留在用的机器人
for r in "${KEEP_ROBOTS[@]}"; do
  FILTERS+=( --include="$ROBOTS_BASE/$r/***" )
done
# 排除其余所有机器人变体
FILTERS+=( --exclude="$ROBOTS_BASE/*" )

# 如需远程镜像同步（删掉远程多余的文件），取消下面这行注释
# FILTERS+=( --delete )

# 空跑模式：DRY_RUN=1 时只列清单，不实际传输
DRY_RUN_FLAGS=""
[[ "${DRY_RUN:-0}" == "1" ]] && DRY_RUN_FLAGS="-n -v"

echo "==> $LOCAL_SRC  →  $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
[[ -n "$DRY_RUN_FLAGS" ]] && echo "==> 空跑模式（DRY_RUN=1），只列清单不传输"

rsync -azh --partial --info=progress2 \
  $DRY_RUN_FLAGS \
  "${FILTERS[@]}" \
  "$LOCAL_SRC" \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

echo "==> 同步完成"
