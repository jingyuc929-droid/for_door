#!/usr/bin/env bash
set -euo pipefail

# Closed-loop evaluation of the distilled recurrent student. All reset and
# dynamics variables accepted by test.sh can also be supplied here.
#
# Examples:
#   scripts/test_student.sh
#   STUDENT_SENSOR_PROFILE=clean scripts/test_student.sh
#   EVAL_EPISODES=300 SEEDS="42 43 44" scripts/test_student.sh
#   VIDEO=1 NUM_ENVS=1 EVAL_EPISODES=1 SEEDS=42 scripts/test_student.sh

STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-logs/student/dagger_student/student_best.pt}"
STUDENT_SENSOR_PROFILE="${STUDENT_SENSOR_PROFILE:-checkpoint}"

if [[ ! -f "${STUDENT_CHECKPOINT}" ]]; then
    echo "[ERROR] Student checkpoint not found: ${STUDENT_CHECKPOINT}" >&2
    exit 2
fi

sensor_args=(--student_sensor_profile "${STUDENT_SENSOR_PROFILE}")
for item in \
    PROPRIO_DELAY_STEPS:proprio_delay_steps \
    GEOMETRY_DELAY_STEPS:geometry_delay_steps \
    PANEL_DELAY_STEPS:panel_delay_steps \
    ARM_POS_NOISE:arm_pos_noise \
    ARM_VEL_NOISE:arm_vel_noise \
    IMU_ANG_VEL_NOISE:imu_ang_vel_noise \
    GRAVITY_NOISE:gravity_noise \
    BASE_HEIGHT_NOISE:base_height_noise \
    DOORWAY_POSITION_NOISE:doorway_position_noise \
    HANDLE_POSITION_NOISE:handle_position_noise \
    DIRECTION_NOISE_DEG:direction_noise_deg \
    PANEL_DIRECTION_NOISE_DEG:panel_direction_noise_deg \
    PANEL_DROPOUT_PROB:panel_dropout_prob
do
    env_name="${item%%:*}"
    arg_name="${item#*:}"
    value="${!env_name:-}"
    if [[ -n "${value}" ]]; then
        sensor_args+=("--${arg_name}" "${value}")
    fi
done

EVAL_SCRIPT=scripts/rsl_rl/eval_student.py \
TEST_NAME="${TEST_NAME:-student_baseline}" \
scripts/test.sh --student_checkpoint "${STUDENT_CHECKPOINT}" "${sensor_args[@]}" "$@"
