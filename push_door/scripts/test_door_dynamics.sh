#!/usr/bin/env bash
set -euo pipefail

# Single-variable sweeps for door and handle actuator dynamics. Each run keeps
# every scale at 1.0 except the selected variable.
#
# Quick examples:
#   VARIABLES="door_friction" LEVELS="0.5 1.0 1.5 2.0" scripts/test_door_dynamics.sh
#   VARIABLES="handle_damping" LEVELS="0.5 1.0 2.0" scripts/test_door_dynamics.sh
# Formal run:
#   SEEDS="42 43 44" EVAL_EPISODES=300 scripts/test_door_dynamics.sh

VARIABLES="${VARIABLES:-door_friction door_damping door_stiffness handle_friction handle_damping handle_stiffness}"
LEVELS="${LEVELS:-0.50 0.75 1.00 1.25 1.50 2.00}"
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
        door_stiffness) scale_env="DOOR_STIFFNESS_SCALE" ;;
        door_damping) scale_env="DOOR_DAMPING_SCALE" ;;
        door_friction) scale_env="DOOR_FRICTION_SCALE" ;;
        handle_stiffness) scale_env="HANDLE_STIFFNESS_SCALE" ;;
        handle_damping) scale_env="HANDLE_DAMPING_SCALE" ;;
        handle_friction) scale_env="HANDLE_FRICTION_SCALE" ;;
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

        echo "[INFO] Starting dynamics test ${variable}=${level}x"
        env "${env_args[@]}" scripts/test.sh
    done
done

echo "[INFO] Door/handle dynamics sweeps completed. Results are grouped under logs/eval/."
