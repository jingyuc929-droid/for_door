#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   DOOR_MODE=push scripts/play.sh
#   DOOR_MODE=pull CHECKPOINT=logs/rsl_rl/door_pull_asymmetric_critic/.../model.pt scripts/play.sh

DOOR_MODE="${DOOR_MODE:-push}"
case "${DOOR_MODE}" in
    push)
        DEFAULT_TASK="Template-Door-Env-v0"
        DEFAULT_EXPERIMENT_NAME="door_asymmetric_critic"
        DEFAULT_CHECKPOINT="logs/rsl_rl/door_asymmetric_critic/2026-07-21_14-02-59_normal/model_1300.pt"
        ;;
    pull)
        DEFAULT_TASK="Template-Pull-Door-Env-v0"
        DEFAULT_EXPERIMENT_NAME="door_pull_asymmetric_critic"
        DEFAULT_CHECKPOINT="logs/rsl_rl/door_asymmetric_critic/2026-07-23_17-25-55_normal/model_400.pt"
        ;;
    *)
        echo "[ERROR] DOOR_MODE must be push or pull; got: ${DOOR_MODE}" >&2
        exit 2
        ;;
esac

TASK="${TASK:-${DEFAULT_TASK}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
AGENT="${AGENT:-rsl_rl_teacher_cfg_entry_point}"
NUM_ENVS="${NUM_ENVS:-1}"
STATS_EVERY="${STATS_EVERY:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/jing/anaconda3/envs/isaac/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ -z "${CHECKPOINT}" ]]; then
    echo "[ERROR] CHECKPOINT is required for DOOR_MODE=${DOOR_MODE}." >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi

echo "[INFO] door_mode=${DOOR_MODE} task=${TASK} experiment=${EXPERIMENT_NAME} checkpoint=${CHECKPOINT}"
exec "${PYTHON_BIN}" scripts/rsl_rl/play.py \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --stats_every "${STATS_EVERY}" \
    --checkpoint "${CHECKPOINT}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --agent "${AGENT}" \
    "$@"
