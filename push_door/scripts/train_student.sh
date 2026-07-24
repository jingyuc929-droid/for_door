#!/usr/bin/env bash
set -euo pipefail

# Minimal GRU Student overfit/training entry point.
# Example:
#   DATASET=logs/distillation/2026-07-23_xxx_teacher_rollout scripts/train_student.sh
#   DATASETS="logs/distillation/easy logs/distillation/medium" scripts/train_student.sh

DATASET="${DATASET:-}"
DATASETS="${DATASETS:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/jing/anaconda3/envs/isaac/bin/python}"
MAX_EPISODES="${MAX_EPISODES:-20}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
MLP_SIZE="${MLP_SIZE:-128}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-}"
LATENT_LOSS_WEIGHT="${LATENT_LOSS_WEIGHT:-0.0}"
NORMALIZE_TEACHER_LATENT="${NORMALIZE_TEACHER_LATENT:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.0}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT="${OUTPUT:-}"

if [[ -z "${DATASET}" && -z "${DATASETS}" ]]; then
    echo "[ERROR] DATASET or DATASETS must point to rollout directorie(s)." >&2
    exit 2
fi

dataset_args=()
if [[ -n "${DATASETS}" ]]; then
    read -r -a dataset_list <<< "${DATASETS}"
    if [[ "${#dataset_list[@]}" -eq 0 ]]; then
        echo "[ERROR] DATASETS must contain at least one rollout directory." >&2
        exit 2
    fi
    dataset_args+=(--datasets)
    for dataset_dir in "${dataset_list[@]}"; do
        if [[ ! -f "${dataset_dir}/metadata.json" || ! -d "${dataset_dir}/chunks" ]]; then
            echo "[ERROR] Invalid rollout dataset: ${dataset_dir}" >&2
            exit 2
        fi
        dataset_args+=("${dataset_dir}")
    done
else
    if [[ ! -f "${DATASET}/metadata.json" || ! -d "${DATASET}/chunks" ]]; then
        echo "[ERROR] Invalid rollout dataset: ${DATASET}" >&2
        exit 2
    fi
    dataset_args+=(--dataset "${DATASET}")
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi

args=(
    scripts/rsl_rl/train_student.py
    "${dataset_args[@]}"
    --max_episodes "${MAX_EPISODES}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --hidden_size "${HIDDEN_SIZE}"
    --mlp_size "${MLP_SIZE}"
    --latent_loss_weight "${LATENT_LOSS_WEIGHT}"
    --val_fraction "${VAL_FRACTION}"
    --device "${DEVICE}"
)
if [[ -n "${TEACHER_CHECKPOINT}" ]]; then
    args+=(--teacher_checkpoint "${TEACHER_CHECKPOINT}")
fi
case "${NORMALIZE_TEACHER_LATENT}" in
    0) args+=(--no-normalize_teacher_latent) ;;
    1) args+=(--normalize_teacher_latent) ;;
    *)
        echo "[ERROR] NORMALIZE_TEACHER_LATENT must be 0 or 1." >&2
        exit 2
        ;;
esac
if [[ -n "${OUTPUT}" ]]; then
    args+=(--output "${OUTPUT}")
fi

exec "${PYTHON_BIN}" "${args[@]}" "$@"
