#!/usr/bin/env bash
set -euo pipefail

# Deterministic full-task evaluation for a trained DoorBot teacher checkpoint.
# An episode ends on traverse success, fall, bad orientation, or the 15 s timeout.
#
# Examples:
#   DOOR_MODE=push scripts/test.sh
#   DOOR_MODE=pull CHECKPOINT=logs/rsl_rl/door_pull_asymmetric_critic/.../model.pt scripts/test.sh
#   SEEDS="42 43 44 45 46" EVAL_EPISODES=500 NUM_ENVS=128 scripts/test.sh
#   CHECKPOINT=logs/rsl_rl/door_asymmetric_critic/my_run/model_8000.pt scripts/test.sh
#   TEST_NAME=xy_010 RESET_XY_RANGE=0.10 RESET_YAW_RANGE=0.40 scripts/test.sh
#   TEST_NAME=yaw_060 RESET_XY_RANGE=0.05 RESET_YAW_RANGE=0.60 scripts/test.sh
#   TEST_NAME=arm_004 ARM_JOINT_POS_RANGE=0.04 RESET_XY_RANGE=0.05 RESET_YAW_RANGE=0.40 scripts/test.sh
#   VIDEO=1 SEEDS=42 NUM_ENVS=1 EVAL_EPISODES=1 scripts/test.sh

DOOR_MODE="${DOOR_MODE:-push}"
case "${DOOR_MODE}" in
    push)
        DEFAULT_TASK="Template-Door-Env-v0"
        DEFAULT_CHECKPOINT="logs/rsl_rl/door_asymmetric_critic/2026-07-21_14-02-59_normal/model_1300.pt"
        ;;
    pull)
        DEFAULT_TASK="Template-Pull-Door-Env-v0"
        DEFAULT_CHECKPOINT=""
        ;;
    *)
        echo "[ERROR] DOOR_MODE must be push or pull; got: ${DOOR_MODE}" >&2
        exit 2
        ;;
esac

TASK="${TASK:-${DEFAULT_TASK}}"
AGENT="${AGENT:-rsl_rl_teacher_cfg_entry_point}"
PYTHON_BIN="${PYTHON_BIN:-/home/jing/anaconda3/envs/isaac/bin/python}"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
EVAL_SCRIPT="${EVAL_SCRIPT:-scripts/rsl_rl/eval_teacher.py}"

NUM_ENVS="${NUM_ENVS:-32}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
SEEDS="${SEEDS:-42 43 44}"
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-750}"
TEST_NAME="${TEST_NAME:-baseline}"
# Empty means: keep the value declared by the task cfg.
RESET_XY_RANGE="${RESET_XY_RANGE:-}"
RESET_YAW_RANGE="${RESET_YAW_RANGE:-}"
ARM_JOINT_POS_RANGE="${ARM_JOINT_POS_RANGE:-}"
DOOR_STIFFNESS_SCALE="${DOOR_STIFFNESS_SCALE:-1.0}"
DOOR_DAMPING_SCALE="${DOOR_DAMPING_SCALE:-1.0}"
DOOR_FRICTION_SCALE="${DOOR_FRICTION_SCALE:-1.0}"
HANDLE_STIFFNESS_SCALE="${HANDLE_STIFFNESS_SCALE:-1.0}"
HANDLE_DAMPING_SCALE="${HANDLE_DAMPING_SCALE:-1.0}"
HANDLE_FRICTION_SCALE="${HANDLE_FRICTION_SCALE:-1.0}"
ARM_EFFORT_SCALE="${ARM_EFFORT_SCALE:-1.0}"
ARM_STIFFNESS_SCALE="${ARM_STIFFNESS_SCALE:-1.0}"
ARM_DAMPING_SCALE="${ARM_DAMPING_SCALE:-1.0}"
ARM_ACTION_SCALE="${ARM_ACTION_SCALE:-1.0}"

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

case "${VIDEO}" in
    0|1) ;;
    *)
        echo "[ERROR] VIDEO must be 0 or 1; got: ${VIDEO}" >&2
        exit 2
        ;;
esac

read -r -a seed_list <<< "${SEEDS}"
if [[ "${#seed_list[@]}" -eq 0 ]]; then
    echo "[ERROR] SEEDS must contain at least one integer seed." >&2
    exit 2
fi

for seed in "${seed_list[@]}"; do
    args=(
        "${EVAL_SCRIPT}"
        --task "${TASK}"
        --agent "${AGENT}"
        --checkpoint "${CHECKPOINT}"
        --num_envs "${NUM_ENVS}"
        --eval_episodes "${EVAL_EPISODES}"
        --seed "${seed}"
        --test_name "${TEST_NAME}"
        --disable_staged_reset true
        --deterministic true
        --door_stiffness_scale "${DOOR_STIFFNESS_SCALE}"
        --door_damping_scale "${DOOR_DAMPING_SCALE}"
        --door_friction_scale "${DOOR_FRICTION_SCALE}"
        --handle_stiffness_scale "${HANDLE_STIFFNESS_SCALE}"
        --handle_damping_scale "${HANDLE_DAMPING_SCALE}"
        --handle_friction_scale "${HANDLE_FRICTION_SCALE}"
        --arm_effort_scale "${ARM_EFFORT_SCALE}"
        --arm_stiffness_scale "${ARM_STIFFNESS_SCALE}"
        --arm_damping_scale "${ARM_DAMPING_SCALE}"
        --arm_action_scale "${ARM_ACTION_SCALE}"
        --headless
    )

    if [[ "${VIDEO}" == "1" ]]; then
        args+=(--video --video_length "${VIDEO_LENGTH}")
    fi
    if [[ -n "${RESET_XY_RANGE}" ]]; then
        args+=(--reset_xy_range "${RESET_XY_RANGE}")
    fi
    if [[ -n "${RESET_YAW_RANGE}" ]]; then
        args+=(--reset_yaw_range "${RESET_YAW_RANGE}")
    fi
    if [[ -n "${ARM_JOINT_POS_RANGE}" ]]; then
        args+=(--arm_joint_pos_range "${ARM_JOINT_POS_RANGE}")
    fi

    echo "[INFO] test=${TEST_NAME} door_mode=${DOOR_MODE} task=${TASK} checkpoint=${CHECKPOINT} seed=${seed} envs=${NUM_ENVS} episodes=${EVAL_EPISODES} reset_xy=${RESET_XY_RANGE:-task_cfg} reset_yaw=${RESET_YAW_RANGE:-task_cfg} arm_joint_pos=${ARM_JOINT_POS_RANGE:-0.0} door_scales=[k:${DOOR_STIFFNESS_SCALE},d:${DOOR_DAMPING_SCALE},f:${DOOR_FRICTION_SCALE}] handle_scales=[k:${HANDLE_STIFFNESS_SCALE},d:${HANDLE_DAMPING_SCALE},f:${HANDLE_FRICTION_SCALE}] arm_actuator=[effort:${ARM_EFFORT_SCALE},k:${ARM_STIFFNESS_SCALE},d:${ARM_DAMPING_SCALE},action:${ARM_ACTION_SCALE}]"
    "${PYTHON_BIN}" "${args[@]}" "$@"
done

echo "[INFO] All evaluation seeds completed. Results are under logs/eval/${TEST_NAME}/."
