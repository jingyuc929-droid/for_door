#!/usr/bin/env bash
set -euo pipefail

# Single-variable sweep of the six arm joints' initial position offsets.
# The quadruped joints are always reset to their nominal pose.
#
# Examples:
#   scripts/test_arm_joint.sh
#   LEVELS="0.00 0.02 0.04" EVAL_EPISODES=50 SEEDS=42 scripts/test_arm_joint.sh
#   CHECKPOINT=logs/rsl_rl/door_asymmetric_critic/my_run/model_1300.pt scripts/test_arm_joint.sh

LEVELS="${LEVELS:-0.00 0.02 0.04 0.06 0.10}"
RESET_XY_RANGE="${RESET_XY_RANGE:-0.05}"
RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.40}"
SEEDS="${SEEDS:-42 43 44}"
NUM_ENVS="${NUM_ENVS:-32}"
EVAL_EPISODES="${EVAL_EPISODES:-300}"
DOOR_MODE="${DOOR_MODE:-push}"
CHECKPOINT="${CHECKPOINT:-}"

read -r -a level_list <<< "${LEVELS}"
if [[ "${#level_list[@]}" -eq 0 ]]; then
    echo "[ERROR] LEVELS must contain at least one non-negative range." >&2
    exit 2
fi

for level in "${level_list[@]}"; do
    label="${level//./}"
    env_args=(
        "DOOR_MODE=${DOOR_MODE}"
        "TEST_NAME=arm_${label}"
        "ARM_JOINT_POS_RANGE=${level}"
        "RESET_XY_RANGE=${RESET_XY_RANGE}"
        "RESET_YAW_RANGE=${RESET_YAW_RANGE}"
        "SEEDS=${SEEDS}"
        "NUM_ENVS=${NUM_ENVS}"
        "EVAL_EPISODES=${EVAL_EPISODES}"
    )
    if [[ -n "${CHECKPOINT}" ]]; then
        env_args+=("CHECKPOINT=${CHECKPOINT}")
    fi

    echo "[INFO] Starting arm-joint test range=±${level} rad"
    env "${env_args[@]}" scripts/test.sh
done

echo "[INFO] Arm-joint single-variable sweep completed. Results are under logs/eval/arm_*/."
