# DoorBot deployment bundle

This repository contains the two source trees required by the door-opening project:

- `push_door/`: door task, training/evaluation scripts, robot and door assets.
- `pick_and_place_yzc/`: the original Pick & Place project providing `rl_sim_env`,
  `rl_algorithms`, locomotion configuration, robot descriptions, and assets.
- `wbc_szy/`: the WBC source tree copied from `/home/jing/wbc_szy`.
- `checkpoints/`: selected deployable teacher and student policies.

Large training logs, rollout datasets, videos, caches, backup archives, and the
original Git metadata are intentionally excluded.

## Environment

Install a compatible NVIDIA driver, CUDA, Isaac Sim, Isaac Lab, and PyTorch first.
Use the same Isaac Sim/Isaac Lab version on which the policies were trained.

Install both Python extensions from this repository:

```bash
python -m pip install -e pick_and_place_yzc/source/rl_sim_env
python -m pip install -e push_door/source/door_env
```

Set the external low-level source location before running DoorBot:

```bash
export DOORBOT_LOW_LEVEL_RL_SOURCE_ROOT="$PWD/pick_and_place_yzc/source/rl_sim_env"
```

The launch scripts currently accept `PYTHON_BIN` and checkpoint paths as
environment variables. For example:

```bash
cd push_door
PYTHON_BIN=/path/to/isaaclab/python \
CHECKPOINT=../checkpoints/teacher/push_model_1300.pt \
DOORBOT_LOW_LEVEL_RL_SOURCE_ROOT="$PWD/../pick_and_place_yzc/source/rl_sim_env" \
scripts/play.sh
```

For pull-door evaluation, set `DOOR_MODE=pull` and use
`../checkpoints/teacher/pull_model_400.pt`.

## Included policies

- `push_door/source/door_env/door_env/tasks/manager_based/door_env/low_level_locomotion/model_30000.pt`:
  frozen low-level locomotion policy used by the hierarchical controller.
- `checkpoints/teacher/push_model_1300.pt`: selected push-door teacher policy.
- `checkpoints/teacher/pull_model_400.pt`: selected pull-door teacher policy.
- `checkpoints/student/student_best.pt`: selected student policy.

The source trees still contain a few historical absolute paths in conversion
metadata and debugging defaults. Runtime scripts should be launched with the
environment variables shown above.
