#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-Template-Door-Env-v0}"
CHECKPOINT="${CHECKPOINT:-/home/jing/push_door/logs/rsl_rl/door_asymmetric_critic/2026-07-21_14-02-59_normal/model_1300.pt}"
NUM_ENVS="${NUM_ENVS:-32}"
EPISODES="${EPISODES:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/jing/anaconda3/envs/isaac/bin/python}"
LOW_LEVEL_RL_SOURCE_ROOT="${LOW_LEVEL_RL_SOURCE_ROOT:-/home/jing/pick_and_place_yzc/source/rl_sim_env}"
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
ARM_DAMPING_SCALE="${ARM_DAMPING_SCALE:-1.0}"
ARM_ACTION_SCALE="${ARM_ACTION_SCALE:-1.0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

modules_dir="${LOW_LEVEL_RL_SOURCE_ROOT}/rl_algorithms/rsl_rl/modules"
if [[ ! -f "${modules_dir}/actor_critic_locomotion.py" || ! -f "${modules_dir}/vae_blind.py" ]]; then
  echo "[ERROR] Low-level policy source files not found under: ${modules_dir}" >&2
  exit 2
fi
export DOORBOT_LOW_LEVEL_RL_SOURCE_ROOT="${LOW_LEVEL_RL_SOURCE_ROOT}"

args=(
  scripts/rsl_rl/collect_teacher_rollouts.py
  --task "${TASK}"
  --checkpoint "${CHECKPOINT}"
  --num_envs "${NUM_ENVS}"
  --episodes "${EPISODES}"
  --deterministic
  --disable_staged_reset
  --door_stiffness_scale "${DOOR_STIFFNESS_SCALE}"
  --door_damping_scale "${DOOR_DAMPING_SCALE}"
  --door_friction_scale "${DOOR_FRICTION_SCALE}"
  --handle_stiffness_scale "${HANDLE_STIFFNESS_SCALE}"
  --handle_damping_scale "${HANDLE_DAMPING_SCALE}"
  --handle_friction_scale "${HANDLE_FRICTION_SCALE}"
  --arm_effort_scale "${ARM_EFFORT_SCALE}"
  --arm_damping_scale "${ARM_DAMPING_SCALE}"
  --arm_action_scale "${ARM_ACTION_SCALE}"
  --proprio_delay_steps "${PROPRIO_DELAY_STEPS:-0}"
  --geometry_delay_steps "${GEOMETRY_DELAY_STEPS:-2}"
  --panel_delay_steps "${PANEL_DELAY_STEPS:-4}"
  --panel_direction_noise_deg "${PANEL_DIRECTION_NOISE_DEG:-10.0}"
  --panel_dropout_prob "${PANEL_DROPOUT_PROB:-0.08}"
  --headless
)
if [[ -n "${RESET_XY_RANGE}" ]]; then
  args+=(--reset_xy_range "${RESET_XY_RANGE}")
fi
if [[ -n "${RESET_YAW_RANGE}" ]]; then
  args+=(--reset_yaw_range "${RESET_YAW_RANGE}")
fi
if [[ -n "${ARM_JOINT_POS_RANGE}" ]]; then
  args+=(--arm_joint_pos_range "${ARM_JOINT_POS_RANGE}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  args+=(--output_dir "${OUTPUT_DIR}")
fi

echo "[INFO] teacher rollout randomization reset_xy=${RESET_XY_RANGE:-task_cfg} reset_yaw=${RESET_YAW_RANGE:-task_cfg} arm_joint_pos=${ARM_JOINT_POS_RANGE:-0.0} door=[k:${DOOR_STIFFNESS_SCALE},d:${DOOR_DAMPING_SCALE},f:${DOOR_FRICTION_SCALE}] handle=[k:${HANDLE_STIFFNESS_SCALE},d:${HANDLE_DAMPING_SCALE},f:${HANDLE_FRICTION_SCALE}] arm=[effort:${ARM_EFFORT_SCALE},damping:${ARM_DAMPING_SCALE},action:${ARM_ACTION_SCALE}]"
"${PYTHON_BIN}" "${args[@]}" "$@"
