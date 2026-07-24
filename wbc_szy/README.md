# RL SIM ENV

## Submodule
```bash
git submodule update --init --recursive
```

## Ldd Check
```bash
ldd --version # GLIBC 2.34+ version
```

## NVIDIA Driver
```bash
570.169
```

## Conda Environment Setup
```bash
conda create -n [env_name] python=3.11
conda activate [env_name]
conda install onnxruntime
pip install --upgrade pip
pip install ruamel.yaml
pip install wandb==0.18.7
# pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install tensordict
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

# verifying the isaac sim
isaacsim

# in other path !!!!
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 3a1a65bd942121b059ca4356819c8353a6840af8
./isaaclab.sh --install

# in rl_sim_env path !!!!
python -m pip install -e source/rl_sim_env
```

## project list
* locomotion
    * grq20_v2d4_x5_default: whole-body locomotion for GRQ20 V2D4 with the X5 arm.
    * grq20_v2d4_x5_terrain: whole-body locomotion on terrain for GRQ20 V2D4 with the X5 arm.
    * grq20_v2d4_x5_smooth_terrain: whole-body locomotion on smooth terrain for GRQ20 V2D4 with the X5 arm.
    * grq20_v2d4_x5_force_control: whole-body force-control training for GRQ20 V2D4 with the X5 arm.

## Wandb Setup

Sign up to wandb: https://wandb.ai/ and get the API key.

Run the following command to login to wandb:
```bash

wandb login
export WANDB_USERNAME=USER_NAME
```

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Show Available Environments
```bash
python scripts/list_envs.py
```

## Train

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### single-gpu
```bash
python scripts/[env_name]/train.py --task [task_name] --run_name [run_name] --headless --kit_args="--/physics/collisionApproximateCylinders=true"
```
### multi-gpu
```bash
export JAX_LOCAL_RANK=[start_gpu_id]
python -m torch.distributed.run --nnodes=[server_num] --nproc_per_node=[gpu_num] --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29502 scripts/[env_name]/train.py --task [task_name] --run_name [run_name] --device cuda:[start_gpu_id] --headless --distributed
```