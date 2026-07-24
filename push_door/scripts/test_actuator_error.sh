#!/usr/bin/env bash
set -euo pipefail

# Single-variable sweep for arm actuator/controller calibration errors. Every
# run keeps all scales at 1.0 except the selected variable.
#
# Examples:
#   scripts/test_actuator_error.sh
#   VARIABLES="arm_effort" LEVELS="0.7 0.85 1.0" scripts/test_actuator_error.sh
#   SEEDS="42 43 44" EVAL_EPISODES=300 scripts/test_actuator_error.sh

VARIABLES="${VARIABLES:-arm_effort arm_stiffness arm_damping arm_action}"
LEVELS="${LEVELS:-0.70 0.85 1.00 1.15 1.30}"
RESET_XY_RANGE="${RESET_XY_RANGE:-0.05}"
RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.40}"
SEEDS="${SEEDS:-42}"
NUM_ENVS="${NUM_ENVS:-32}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
DOOR_MODE="${DOOR_MODE:-push}"
CHECKPOINT="${CHECKPOINT:-}"

read -r -a variable_list <<< "${VARIABLES}"
read -r -a level_list <<< "${LEVELS}"
if [[ "${#variable_list[@]}" -eq 0 || "${#level_list[@]}" -eq 0 ]]; then
    echo "[ERROR] VARIABLES and LEVELS must both be non-empty." >&2
    exit 2
fi

for variable in "${variable_list[@]}"; do
    case "${variable}" in
        arm_effort) scale_env="ARM_EFFORT_SCALE" ;;
        arm_stiffness) scale_env="ARM_STIFFNESS_SCALE" ;;
        arm_damping) scale_env="ARM_DAMPING_SCALE" ;;
        arm_action) scale_env="ARM_ACTION_SCALE" ;;
        *)
            echo "[ERROR] Unsupported variable: ${variable}" >&2
            exit 2
            ;;
    esac

    for level in "${level_list[@]}"; do
        label="${level//./}"
        env_args=(
            "DOOR_MODE=${DOOR_MODE}"
            "TEST_NAME=${variable}_${label}x"
            "RESET_XY_RANGE=${RESET_XY_RANGE}"
            "RESET_YAW_RANGE=${RESET_YAW_RANGE}"
            "SEEDS=${SEEDS}"
            "NUM_ENVS=${NUM_ENVS}"
            "EVAL_EPISODES=${EVAL_EPISODES}"
            "${scale_env}=${level}"
        )
        if [[ -n "${CHECKPOINT}" ]]; then
            env_args+=("CHECKPOINT=${CHECKPOINT}")
        fi

        echo "[INFO] Starting actuator-error test ${variable}=${level}x"
        env "${env_args[@]}" scripts/test.sh
    done
done

echo "[INFO] Arm actuator-error sweeps completed. Results are grouped under logs/eval/."
