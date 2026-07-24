#!/usr/bin/env bash
# Pull the newest RobotDebugData capture from the robot, convert it to a CSV
# understood by robot_data_plot, and point the plot launcher at that CSV.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-galileo-dog}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-/home/galileo/dev_arm/control/grq20_v2d4_piperL/release/launch/logs}"
PLOT_ROOT="${PLOT_ROOT:-/home/sun/galileo_sim/robot_data_plot}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OPEN_PLOT=false

usage() {
    cat <<'EOF'
用法: ./pull_robot_plot_data.sh [--plot]

自动完成：
  1. 从机器人端选择最新的 RobotDebugData 日志；
  2. 按日期拷到 robot_data_plot/logs；
  3. 转换为 plot_robot_data.py 可读取的 CSV；
  4. 更新 scripts/log_path.env，使 plot 启动脚本指向该 CSV。

选项：
  --plot    完成后立即启动绘图界面
  -h, --help

可通过环境变量覆盖 REMOTE_HOST、REMOTE_LOG_ROOT、PLOT_ROOT、PYTHON_BIN。
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plot)
            OPEN_PLOT=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "错误: 未知参数: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

converter="$PLOT_ROOT/lcm/diagnose_piper_lcm.py"
plot_launcher="$PLOT_ROOT/scripts/plot_robot_data.launch.sh"
plot_config="$PLOT_ROOT/scripts/log_path.env"

for required_file in "$converter" "$plot_launcher"; do
    if [[ ! -f "$required_file" ]]; then
        echo "错误: 文件不存在: $required_file" >&2
        exit 1
    fi
done

# The deployed recorder creates chronological YYYY-MM-DD/HH-MM-SS names.
# Compression can be interrupted by a power loss, leaving both the intact
# append-only .log and an unusable partial .log.xz.  Select the newest capture
# across all supported formats, but prefer its raw .log when both exist.
remote_files="$({
    ssh -o BatchMode=yes "$REMOTE_HOST" \
        "find '$REMOTE_LOG_ROOT' -mindepth 2 -maxdepth 2 -type f \\
          \( -name '*.log' -o -name '*.log.xz' -o -name '*.log.gz' \\
             -o -name '*.log.bz2' -o -name '*.log.lzma' \) \\
          -print 2>/dev/null"
} || true)"

remote_file=""
remote_key=""
remote_is_raw=0
while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    candidate_date="$(basename "$(dirname "$candidate")")"
    candidate_name="$(basename "$candidate")"
    candidate_time="${candidate_name%%.log*}"
    if [[ ! "$candidate_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || \
       [[ ! "$candidate_time" =~ ^[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
        continue
    fi
    candidate_key="$candidate_date/$candidate_time"
    candidate_is_raw=0
    [[ "$candidate_name" == *.log ]] && candidate_is_raw=1
    if [[ -z "$remote_file" || "$candidate_key" > "$remote_key" || \
          ( "$candidate_key" == "$remote_key" && \
            "$candidate_is_raw" -gt "$remote_is_raw" ) ]]; then
        remote_file="$candidate"
        remote_key="$candidate_key"
        remote_is_raw="$candidate_is_raw"
    fi
done <<< "$remote_files"

if [[ -z "$remote_file" ]]; then
    echo "错误: $REMOTE_HOST:$REMOTE_LOG_ROOT 下没有找到日志" >&2
    exit 1
fi

log_date="$(basename "$(dirname "$remote_file")")"
log_name="$(basename "$remote_file")"
log_time="${log_name%%.log*}"
if [[ ! "$log_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || \
   [[ ! "$log_time" =~ ^[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "错误: 无法从远端路径解析日期和时间: $remote_file" >&2
    exit 1
fi

local_log_dir="$PLOT_ROOT/logs/$log_date"
local_log="$local_log_dir/$log_name"
local_csv="$local_log_dir/$log_time.csv"
mkdir -p "$local_log_dir"

echo "==> 最新远端日志: $REMOTE_HOST:$remote_file"
echo "==> 回传到: $local_log"
copy_tmp="$(mktemp "$local_log.tmp.XXXXXX")"
cleanup_copy() {
    rm -f "$copy_tmp"
}
trap cleanup_copy EXIT
scp -p "$REMOTE_HOST:$remote_file" "$copy_tmp"
mv -f "$copy_tmp" "$local_log"
trap - EXIT

if [[ ! -s "$local_log" ]]; then
    echo "错误: 回传后的日志为空: $local_log" >&2
    exit 1
fi

echo "==> 生成绘图 CSV: $local_csv"
local_packages="$PLOT_ROOT/.python_packages"
if [[ -d "$local_packages" ]]; then
    export PYTHONPATH="$local_packages${PYTHONPATH:+:$PYTHONPATH}"
fi
"$PYTHON_BIN" "$converter" --csv-dir "$local_log_dir" "$local_log"

if [[ ! -s "$local_csv" ]]; then
    echo "错误: CSV 未成功生成: $local_csv" >&2
    exit 1
fi

# Only switch the launcher after the copy and conversion have both succeeded.
config_tmp="$(mktemp "$plot_config.tmp.XXXXXX")"
cleanup() {
    rm -f "$config_tmp"
}
trap cleanup EXIT
{
    echo "# 由 pull_robot_plot_data.sh 自动生成；plot 启动脚本默认读取此文件。"
    printf 'LOG_DATE=%q\n' "$log_date"
    printf 'LOG_TIME=%q\n' "$log_time"
    printf 'CSV_FILE=%q\n' "$local_csv"
} > "$config_tmp"
chmod --reference="$plot_config" "$config_tmp" 2>/dev/null || true
mv -f "$config_tmp" "$plot_config"
trap - EXIT

echo "==> 已更新绘图数据指向: $plot_config"
echo "==> 完成。启动命令: $plot_launcher"

if [[ "$OPEN_PLOT" == true ]]; then
    exec env PYTHON_BIN="$PYTHON_BIN" "$plot_launcher"
fi
