#!/usr/bin/env bash
set -euo pipefail

# Combined held-out stress profiles for the push/pull teacher. Unlike the
# single-variable sweeps, each profile changes geometry, arm reset, door/handle
# dynamics, and arm actuator calibration together.
#
# Quick run:
#   scripts/test_combined_stress.sh
# Formal run:
#   SEEDS="42 43 44" EVAL_EPISODES=300 scripts/test_combined_stress.sh
# Selected profiles:
#   PROFILES="baseline medium hard" scripts/test_combined_stress.sh

PROFILES="${PROFILES:-baseline mild medium hard low_resistance}"
SEEDS="${SEEDS:-42}"
NUM_ENVS="${NUM_ENVS:-32}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
DOOR_MODE="${DOOR_MODE:-push}"
CHECKPOINT="${CHECKPOINT:-}"

read -r -a profile_list <<< "${PROFILES}"
if [[ "${#profile_list[@]}" -eq 0 ]]; then
    echo "[ERROR] PROFILES must contain at least one profile." >&2
    exit 2
fi

for profile in "${profile_list[@]}"; do
    # Defaults are the nominal push-task configuration.
    reset_xy="0.05"
    reset_yaw="0.40"
    arm_joint_pos="0.00"
    door_stiffness="1.00"
    door_damping="1.00"
    door_friction="1.00"
    handle_stiffness="1.00"
    handle_damping="1.00"
    handle_friction="1.00"
    arm_effort="1.00"
    arm_stiffness="1.00"
    arm_damping="1.00"
    arm_action="1.00"

    case "${profile}" in
        baseline)
            ;;
        mild)
            reset_xy="0.10"
            reset_yaw="0.50"
            arm_joint_pos="0.02"
            door_stiffness="1.05"
            door_damping="1.10"
            door_friction="1.15"
            handle_stiffness="1.05"
            handle_damping="1.10"
            handle_friction="1.15"
            arm_effort="0.95"
            arm_stiffness="0.95"
            arm_damping="1.05"
            arm_action="0.95"
            ;;
        medium)
            reset_xy="0.15"
            reset_yaw="0.60"
            arm_joint_pos="0.04"
            door_stiffness="1.10"
            door_damping="1.20"
            door_friction="1.30"
            handle_stiffness="1.10"
            handle_damping="1.20"
            handle_friction="1.30"
            arm_effort="0.90"
            arm_stiffness="0.90"
            arm_damping="1.10"
            arm_action="0.90"
            ;;
        hard)
            reset_xy="0.20"
            reset_yaw="0.80"
            arm_joint_pos="0.06"
            door_stiffness="1.20"
            door_damping="1.25"
            door_friction="1.50"
            handle_stiffness="1.20"
            handle_damping="1.25"
            handle_friction="1.50"
            arm_effort="0.80"
            arm_stiffness="0.85"
            arm_damping="1.20"
            arm_action="0.85"
            ;;
        low_resistance)
            # Opposite dynamics corner: the door/handle move more freely and
            # can expose overshoot or premature hook-release behavior.
            reset_xy="0.15"
            reset_yaw="0.60"
            arm_joint_pos="0.04"
            door_stiffness="0.80"
            door_damping="0.70"
            door_friction="0.60"
            handle_stiffness="0.80"
            handle_damping="0.70"
            handle_friction="0.60"
            arm_effort="1.10"
            arm_stiffness="1.10"
            arm_damping="0.90"
            arm_action="1.10"
            ;;
        *)
            echo "[ERROR] Unsupported profile: ${profile}" >&2
            echo "[ERROR] Available profiles: baseline mild medium hard low_resistance" >&2
            exit 2
            ;;
    esac

    env_args=(
        "DOOR_MODE=${DOOR_MODE}"
        "TEST_NAME=stress_${profile}"
        "RESET_XY_RANGE=${reset_xy}"
        "RESET_YAW_RANGE=${reset_yaw}"
        "ARM_JOINT_POS_RANGE=${arm_joint_pos}"
        "DOOR_STIFFNESS_SCALE=${door_stiffness}"
        "DOOR_DAMPING_SCALE=${door_damping}"
        "DOOR_FRICTION_SCALE=${door_friction}"
        "HANDLE_STIFFNESS_SCALE=${handle_stiffness}"
        "HANDLE_DAMPING_SCALE=${handle_damping}"
        "HANDLE_FRICTION_SCALE=${handle_friction}"
        "ARM_EFFORT_SCALE=${arm_effort}"
        "ARM_STIFFNESS_SCALE=${arm_stiffness}"
        "ARM_DAMPING_SCALE=${arm_damping}"
        "ARM_ACTION_SCALE=${arm_action}"
        "SEEDS=${SEEDS}"
        "NUM_ENVS=${NUM_ENVS}"
        "EVAL_EPISODES=${EVAL_EPISODES}"
    )
    if [[ -n "${CHECKPOINT}" ]]; then
        env_args+=("CHECKPOINT=${CHECKPOINT}")
    fi

    echo "[INFO] Starting combined stress profile=${profile}"
    echo "[INFO] reset=[xy:±${reset_xy}m,yaw:±${reset_yaw}rad,arm:±${arm_joint_pos}rad]"
    echo "[INFO] door=[k:${door_stiffness},d:${door_damping},f:${door_friction}] handle=[k:${handle_stiffness},d:${handle_damping},f:${handle_friction}]"
    echo "[INFO] arm_actuator=[effort:${arm_effort},k:${arm_stiffness},d:${arm_damping},action:${arm_action}]"
    env "${env_args[@]}" scripts/test.sh
done

echo "[INFO] Combined stress tests completed. Results are under logs/eval/stress_*/."
