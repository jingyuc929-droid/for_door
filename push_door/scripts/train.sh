#!/usr/bin/env bash
set -euo pipefail

# Piper hook door-opening teacher training / checkpoint fine-tuning.
#
# Default: warm-start a new run from the trained reference checkpoint. This is
# the recommended mode after changing rewards or terminations: model weights and
# observation normalizers are restored, while optimizer state and iteration are
# reset.
#
# Examples:
#   # Use the defaults below (fine-tune model_5000.pt):
#   DOOR_MODE=push scripts/train.sh
#   DOOR_MODE=pull RESUME=0 scripts/train.sh
#
#   # Start from scratch:
#   RESUME=0 RUN_NAME=from_scratch_v1 scripts/train.sh
#
#   # Restore model, optimizer and iteration for an unchanged task:
#   RESUME_MODE=full RUN_NAME=exact_resume_v1 scripts/train.sh
#
#   # Override normal training arguments:
#   NUM_ENVS=256 MAX_ITERATIONS=3000 scripts/train.sh --seed 7

DOOR_MODE="${DOOR_MODE:-push}"
case "${DOOR_MODE}" in
    push)
        DEFAULT_TASK="Template-Door-Env-v0"
        DEFAULT_EXPERIMENT_NAME="door_asymmetric_critic"
        DEFAULT_RESUME="1"
        ;;
    pull)
        DEFAULT_TASK="Template-Pull-Door-Env-v0"
        DEFAULT_EXPERIMENT_NAME="door_pull_asymmetric_critic"
        DEFAULT_RESUME="0"
        ;;
    *)
        echo "[ERROR] DOOR_MODE must be push or pull; got: ${DOOR_MODE}" >&2
        exit 2
        ;;
esac

TASK="${TASK:-${DEFAULT_TASK}}"
AGENT="${AGENT:-rsl_rl_teacher_cfg_entry_point}"
NUM_ENVS="${NUM_ENVS:-3072}"
MAX_ITERATIONS="${MAX_ITERATIONS:-8000}"
RUN_NAME="${RUN_NAME:-normal}"
CONSOLE_LOG_MODE="${DOORBOT_CONSOLE_LOG_MODE:-stage_and_termination}"
PYTHON_BIN="${PYTHON_BIN:-/home/jing/anaconda3/envs/isaac/bin/python}"

# Checkpoint loading. LOAD_RUN is a directory under
# logs/rsl_rl/${EXPERIMENT_NAME}; CHECKPOINT is the file inside that directory.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
RESUME="${RESUME:-${DEFAULT_RESUME}}"
RESUME_MODE="${RESUME_MODE:-finetune}"
LOAD_RUN="${LOAD_RUN:-2026-07-21_12-00-45_normal}"
CHECKPOINT="${CHECKPOINT:-model_1100.pt}"

# Optional training video recording (0/1).
VIDEO="${VIDEO:-0}"
# At 50 Hz policy frequency, 750 steps cover one 15-second episode.
VIDEO_LENGTH="${VIDEO_LENGTH:-750}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-20000}"

export DOORBOT_CONSOLE_LOG_MODE="${CONSOLE_LOG_MODE}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

case "${RESUME}" in
    0|1) ;;
    *)
        echo "[ERROR] RESUME must be 0 or 1; got: ${RESUME}" >&2
        exit 2
        ;;
esac

case "${RESUME_MODE}" in
    full|finetune) ;;
    *)
        echo "[ERROR] RESUME_MODE must be full or finetune; got: ${RESUME_MODE}" >&2
        exit 2
        ;;
esac

args=(
    scripts/rsl_rl/train.py
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --max_iterations "${MAX_ITERATIONS}"
    --agent "${AGENT}"
    --experiment_name "${EXPERIMENT_NAME}"
    --run_name "${RUN_NAME}"
    --headless
)

if [[ "${RESUME}" == "1" ]]; then
    checkpoint_path="logs/rsl_rl/${EXPERIMENT_NAME}/${LOAD_RUN}/${CHECKPOINT}"
    if [[ ! -f "${checkpoint_path}" ]]; then
        echo "[ERROR] Resume checkpoint not found: ${checkpoint_path}" >&2
        exit 3
    fi
    args+=(
        --resume
        --resume_mode "${RESUME_MODE}"
        --load_run "${LOAD_RUN}"
        --checkpoint "${CHECKPOINT}"
    )
fi

if [[ "${VIDEO}" == "1" ]]; then
    args+=(
        --video
        --video_length "${VIDEO_LENGTH}"
        --video_interval "${VIDEO_INTERVAL}"
    )
elif [[ "${VIDEO}" != "0" ]]; then
    echo "[ERROR] VIDEO must be 0 or 1; got: ${VIDEO}" >&2
    exit 2
fi

echo "[INFO] door_mode=${DOOR_MODE} task=${TASK} experiment=${EXPERIMENT_NAME} envs=${NUM_ENVS} iterations=${MAX_ITERATIONS} run=${RUN_NAME}"
if [[ "${RESUME}" == "1" ]]; then
    echo "[INFO] checkpoint=${checkpoint_path} resume_mode=${RESUME_MODE}"
else
    echo "[INFO] checkpoint loading disabled; training from scratch"
fi

# Extra CLI arguments are appended last and can override the defaults above.
exec "${PYTHON_BIN}" "${args[@]}" "$@"
