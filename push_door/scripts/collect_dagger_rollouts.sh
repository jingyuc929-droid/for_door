#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-Template-Door-Env-v0}"
AGENT="${AGENT:-rsl_rl_teacher_cfg_entry_point}"
PYTHON_BIN="${PYTHON_BIN:-/home/jing/anaconda3/envs/isaac/bin/python}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-logs/rsl_rl/door_asymmetric_critic/2026-07-21_14-02-59_normal/model_1300.pt}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-logs/student/student_300/student_best.pt}"
NUM_ENVS="${NUM_ENVS:-32}"
EPISODES="${EPISODES:-300}"
TEACHER_MIX="${TEACHER_MIX:-0.20}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SEED="${SEED:-42}"
VALIDATE_ARGS_ONLY="${VALIDATE_ARGS_ONLY:-0}"

if [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
    echo "[ERROR] Teacher checkpoint not found: ${TEACHER_CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -f "${STUDENT_CHECKPOINT}" ]]; then
    echo "[ERROR] Student checkpoint not found: ${STUDENT_CHECKPOINT}" >&2
    exit 2
fi

args=(
    scripts/rsl_rl/collect_dagger_rollouts.py
    --task "${TASK}"
    --agent "${AGENT}"
    --checkpoint "${TEACHER_CHECKPOINT}"
    --student_checkpoint "${STUDENT_CHECKPOINT}"
    --num_envs "${NUM_ENVS}"
    --episodes "${EPISODES}"
    --teacher_mix "${TEACHER_MIX}"
    --seed "${SEED}"
    --headless
)
if [[ "${VALIDATE_ARGS_ONLY}" == "1" ]]; then
    args+=(--validate_args_only)
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
    args+=(--output_dir "${OUTPUT_DIR}")
fi

echo "[INFO] DAgger teacher=${TEACHER_CHECKPOINT} student=${STUDENT_CHECKPOINT} envs=${NUM_ENVS} episodes=${EPISODES} teacher_mix=${TEACHER_MIX} seed=${SEED}"
"${PYTHON_BIN}" "${args[@]}" "$@"
