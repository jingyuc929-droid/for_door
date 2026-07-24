#!/usr/bin/env python3
"""Train a recurrent DoorBot student from offline teacher rollout chunks."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GRU DoorBot student by teacher action imitation.")
    parser.add_argument("--dataset", type=str, default=None, help="Teacher rollout directory containing metadata.json and chunks/.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="One or more rollout directories. Overrides --dataset when provided.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory; defaults to logs/student/<timestamp>.")
    parser.add_argument("--max_episodes", type=int, default=20, help="Maximum complete episodes to use; 0 means all.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--mlp_size", type=int, default=128)
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Teacher checkpoint used to compute z_priv labels.")
    parser.add_argument("--latent_loss_weight", type=float, default=0.0, help="Weight for privileged latent distillation loss.")
    parser.add_argument("--normalize_teacher_latent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--val_fraction", type=float, default=0.0, help="Episode-level validation fraction; use 0 for overfit test.")
    parser.add_argument("--selection_seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--action_clip",
        type=float,
        default=1.0,
        help="Clip legacy teacher_action_raw targets to the action range; ignored when teacher_action_clipped exists.",
    )
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    if args.datasets is None and args.dataset is None:
        parser.error("Either --dataset or --datasets is required")
    if args.max_episodes < 0:
        parser.error("--max_episodes must be non-negative")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch_size must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning_rate must be positive")
    if args.latent_loss_weight < 0.0:
        parser.error("--latent_loss_weight must be non-negative")
    if args.latent_loss_weight > 0.0 and args.teacher_checkpoint is None:
        parser.error("--teacher_checkpoint is required when --latent_loss_weight > 0")
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("--val_fraction must be in [0, 1)")
    if args.log_interval <= 0:
        parser.error("--log_interval must be positive")
    if args.action_clip <= 0.0:
        parser.error("--action_clip must be positive")
    return args


@dataclass
class Episode:
    episode_id: int
    obs: torch.Tensor
    action: torch.Tensor
    privileged: torch.Tensor | None = None
    dataset_index: int = 0
    source_episode_id: int = 0


def _validate_chunk(payload: dict, path: Path):
    required = ("student_obs", "teacher_action_raw", "done", "episode_id")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"{path} is missing required fields: {missing}")
    t, n = payload["episode_id"].shape[:2]
    expected_prefix = (t, n)
    for key in required:
        if tuple(payload[key].shape[:2]) != expected_prefix:
            raise RuntimeError(
                f"{path}: {key} prefix shape={tuple(payload[key].shape[:2])}, expected={expected_prefix}"
            )
        if torch.is_floating_point(payload[key]) and not torch.isfinite(payload[key]).all():
            raise RuntimeError(f"{path}: {key} contains NaN or Inf")


def load_complete_episodes(dataset_dir: Path, action_clip: float, dataset_index: int = 0) -> tuple[list[Episode], dict]:
    metadata_path = dataset_dir / "metadata.json"
    chunks_dir = dataset_dir / "chunks"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    chunk_paths = sorted(chunks_dir.glob("rollout_*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No rollout_*.pt chunks found under: {chunks_dir}")

    streams: dict[int, dict[str, object]] = {}
    obs_dim = None
    action_dim = None
    for chunk_index, path in enumerate(chunk_paths, start=1):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _validate_chunk(payload, path)
        obs = payload["student_obs"].float()
        if "teacher_action_clipped" in payload:
            action = payload["teacher_action_clipped"].float()
        else:
            action = payload["teacher_action_raw"].float().clamp(-action_clip, action_clip)
        privileged = payload.get("privileged_state")
        if privileged is not None:
            privileged = privileged.float()
        done = payload["done"].bool()
        episode_ids = payload["episode_id"].long()
        obs_dim = obs.shape[-1] if obs_dim is None else obs_dim
        action_dim = action.shape[-1] if action_dim is None else action_dim
        if obs.shape[-1] != obs_dim or action.shape[-1] != action_dim:
            raise RuntimeError(f"Feature dimensions changed in {path}")

        for episode_id_tensor in torch.unique(episode_ids):
            episode_id = int(episode_id_tensor.item())
            mask = episode_ids == episode_id
            stream = streams.setdefault(episode_id, {"obs": [], "action": [], "complete": False})
            stream["obs"].append(obs[mask])
            stream["action"].append(action[mask])
            if privileged is not None:
                stream.setdefault("privileged", []).append(privileged[mask])
            stream["complete"] = bool(stream["complete"]) or bool(done[mask].any().item())
        print(f"[LOAD] {chunk_index}/{len(chunk_paths)} {path.name}")

    episodes = []
    partial_count = 0
    for episode_id in sorted(streams):
        stream = streams[episode_id]
        if not stream["complete"]:
            partial_count += 1
            continue
        episode_obs = torch.cat(stream["obs"], dim=0)
        episode_action = torch.cat(stream["action"], dim=0)
        episode_privileged = torch.cat(stream["privileged"], dim=0) if stream.get("privileged") else None
        episodes.append(Episode(episode_id, episode_obs, episode_action, episode_privileged, dataset_index, episode_id))

    if not episodes:
        raise RuntimeError("No complete episodes were reconstructed from the rollout chunks.")
    print(
        f"[DATA] complete_episodes={len(episodes)} partial_episodes_dropped={partial_count} "
        f"obs_dim={obs_dim} action_dim={action_dim}"
    )
    return episodes, metadata


def load_datasets(dataset_dirs: list[Path], action_clip: float) -> tuple[list[Episode], dict]:
    all_episodes = []
    metadata_items = []
    expected_obs_dim = None
    expected_action_dim = None
    next_episode_id = 0
    for dataset_index, dataset_dir in enumerate(dataset_dirs):
        episodes, metadata = load_complete_episodes(dataset_dir, action_clip, dataset_index)
        for episode in episodes:
            if expected_obs_dim is None:
                expected_obs_dim = episode.obs.shape[-1]
                expected_action_dim = episode.action.shape[-1]
            if episode.obs.shape[-1] != expected_obs_dim or episode.action.shape[-1] != expected_action_dim:
                raise RuntimeError(
                    f"{dataset_dir}: episode dims obs={episode.obs.shape[-1]} action={episode.action.shape[-1]} "
                    f"do not match expected obs={expected_obs_dim} action={expected_action_dim}"
                )
            episode.episode_id = next_episode_id
            next_episode_id += 1
        all_episodes.extend(episodes)
        metadata_items.append({"path": str(dataset_dir), "metadata": metadata, "complete_episodes": len(episodes)})
    if not all_episodes:
        raise RuntimeError("No complete episodes were loaded from the requested datasets.")
    metadata = {
        "format_version": 1,
        "source": "multi_dataset" if len(dataset_dirs) > 1 else "single_dataset",
        "datasets": metadata_items,
        "student_observation_dim": expected_obs_dim,
        "action_dim": expected_action_dim,
    }
    noise_settings = [
        item["metadata"].get("noise_and_delay")
        for item in metadata_items
        if isinstance(item["metadata"].get("noise_and_delay"), dict)
    ]
    if noise_settings and all(settings == noise_settings[0] for settings in noise_settings):
        metadata["noise_and_delay"] = noise_settings[0]
    print(
        f"[DATASETS] count={len(dataset_dirs)} complete_episodes={len(all_episodes)} "
        f"obs_dim={expected_obs_dim} action_dim={expected_action_dim}"
    )
    return all_episodes, metadata


def compute_obs_stats(episodes: list[Episode]) -> tuple[torch.Tensor, torch.Tensor]:
    count = sum(episode.obs.shape[0] for episode in episodes)
    total = sum((episode.obs.double().sum(dim=0) for episode in episodes), start=torch.zeros(episodes[0].obs.shape[-1], dtype=torch.float64))
    total_sq = sum(
        (episode.obs.double().square().sum(dim=0) for episode in episodes),
        start=torch.zeros(episodes[0].obs.shape[-1], dtype=torch.float64),
    )
    mean = total / count
    variance = torch.clamp(total_sq / count - mean.square(), min=1.0e-12)
    return mean.float(), torch.sqrt(variance).float().clamp_min(1.0e-6)


class EpisodeDataset(Dataset):
    def __init__(
        self,
        episodes: list[Episode],
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        teacher_encoder=None,
        latent_mean: torch.Tensor | None = None,
        latent_std: torch.Tensor | None = None,
    ):
        self.episodes = episodes
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.teacher_encoder = teacher_encoder
        self.latent_mean = latent_mean
        self.latent_std = latent_std

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        episode = self.episodes[index]
        obs = (episode.obs - self.obs_mean) / self.obs_std
        if self.teacher_encoder is None:
            latent = torch.empty((episode.obs.shape[0], 0), dtype=episode.obs.dtype)
        else:
            if episode.privileged is None:
                raise RuntimeError("Latent distillation requires privileged_state in every selected episode.")
            with torch.no_grad():
                latent = self.teacher_encoder(episode.privileged)
            if self.latent_mean is not None and self.latent_std is not None:
                latent = (latent - self.latent_mean) / self.latent_std
        return obs, episode.action, latent, episode.episode_id


def collate_episodes(batch):
    observations, actions, latents, episode_ids = zip(*batch)
    lengths = torch.tensor([value.shape[0] for value in observations], dtype=torch.long)
    padded_obs = pad_sequence(observations, batch_first=True)
    padded_actions = pad_sequence(actions, batch_first=True)
    padded_latents = pad_sequence(latents, batch_first=True)
    steps = torch.arange(padded_obs.shape[1]).unsqueeze(0)
    valid_mask = steps < lengths.unsqueeze(1)
    return padded_obs, padded_actions, padded_latents, valid_mask, lengths, torch.tensor(episode_ids, dtype=torch.long)


class StudentGRU(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, mlp_size: int, latent_dim: int = 0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.mlp_size = int(mlp_size)
        self.latent_dim = int(latent_dim)
        self.gru = nn.GRU(self.obs_dim, self.hidden_size, num_layers=1, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.mlp_size),
            nn.ELU(),
            nn.Linear(self.mlp_size, self.action_dim),
        )
        self.latent_head = nn.Linear(self.hidden_size, self.latent_dim) if self.latent_dim > 0 else None

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        features, hidden = self.gru(obs, hidden)
        latent = self.latent_head(features) if self.latent_head is not None else None
        return self.action_head(features), latent, hidden


class TeacherPrivilegedEncoder(nn.Module):
    def __init__(self, checkpoint_path: Path):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("privileged_encoder_state_dict")
        normalizer = checkpoint.get("privileged_normalizer")
        if state_dict is None or normalizer is None:
            raise RuntimeError(f"{checkpoint_path} does not contain privileged encoder metadata.")
        input_dim = int(state_dict["0.weight"].shape[1])
        hidden_0 = int(state_dict["0.weight"].shape[0])
        hidden_1 = int(state_dict["2.weight"].shape[0])
        output_dim = int(state_dict["4.weight"].shape[0])
        self.output_dim = output_dim
        self.mean = normalizer["_mean"].float()
        self.std = normalizer["_std"].float().clamp_min(1.0e-6)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_0),
            nn.ELU(),
            nn.Linear(hidden_0, hidden_1),
            nn.ELU(),
            nn.Linear(hidden_1, output_dim),
        )
        self.encoder.load_state_dict(state_dict, strict=True)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, privileged: torch.Tensor):
        device = privileged.device
        normalized = (privileged - self.mean.to(device)) / self.std.to(device)
        return self.encoder.to(device)(normalized)


def masked_action_mse(prediction, target, valid_mask):
    squared = (prediction - target).square()
    mask = valid_mask.unsqueeze(-1).to(dtype=squared.dtype)
    denominator = valid_mask.sum().clamp_min(1) * prediction.shape[-1]
    loss = (squared * mask).sum() / denominator
    per_dim = (squared * mask).sum(dim=(0, 1)) / valid_mask.sum().clamp_min(1)
    return loss, per_dim


def masked_latent_mse(prediction, target, valid_mask):
    squared = (prediction - target).square()
    mask = valid_mask.unsqueeze(-1).to(dtype=squared.dtype)
    denominator = valid_mask.sum().clamp_min(1) * prediction.shape[-1]
    return (squared * mask).sum() / denominator


@torch.no_grad()
def compute_latent_stats(episodes: list[Episode], teacher_encoder: TeacherPrivilegedEncoder):
    latents = []
    for episode in episodes:
        if episode.privileged is None:
            raise RuntimeError("Latent distillation requires privileged_state in every selected episode.")
        latents.append(teacher_encoder(episode.privileged).cpu())
    values = torch.cat(latents, dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return {
        "mean": values.mean(dim=0),
        "std": std,
        "min": values.min(dim=0).values,
        "max": values.max(dim=0).values,
        "global_mean": float(values.mean().item()),
        "global_std": float(values.std(unbiased=False).item()),
        "global_min": float(values.min().item()),
        "global_max": float(values.max().item()),
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    squared_sum = None
    valid_steps = 0
    latent_loss_sum = 0.0
    latent_dim = int(getattr(model, "latent_dim", 0))
    z_student_sum = torch.zeros(latent_dim, device=device) if latent_dim > 0 else None
    z_teacher_sum = torch.zeros(latent_dim, device=device) if latent_dim > 0 else None
    z_student_sq_sum = torch.zeros(latent_dim, device=device) if latent_dim > 0 else None
    z_teacher_sq_sum = torch.zeros(latent_dim, device=device) if latent_dim > 0 else None
    for obs, actions, latents, mask, _lengths, _ids in loader:
        obs, actions, latents, mask = obs.to(device), actions.to(device), latents.to(device), mask.to(device)
        prediction, z_student, _ = model(obs)
        squared = (prediction - actions).square() * mask.unsqueeze(-1)
        batch_sum = squared.sum(dim=(0, 1))
        squared_sum = batch_sum if squared_sum is None else squared_sum + batch_sum
        if latent_dim > 0 and z_student is not None:
            latent_loss_sum += float(masked_latent_mse(z_student, latents, mask).item()) * int(mask.sum().item())
            z_student_sum += (z_student * mask.unsqueeze(-1)).sum(dim=(0, 1))
            z_teacher_sum += (latents * mask.unsqueeze(-1)).sum(dim=(0, 1))
            z_student_sq_sum += (z_student.square() * mask.unsqueeze(-1)).sum(dim=(0, 1))
            z_teacher_sq_sum += (latents.square() * mask.unsqueeze(-1)).sum(dim=(0, 1))
        valid_steps += int(mask.sum().item())
    per_dim = squared_sum / max(valid_steps, 1)
    latent_mse = latent_loss_sum / max(valid_steps, 1) if latent_dim > 0 else 0.0
    z_student_mean = z_student_sum / max(valid_steps, 1) if latent_dim > 0 else torch.empty(0)
    z_teacher_mean = z_teacher_sum / max(valid_steps, 1) if latent_dim > 0 else torch.empty(0)
    z_student_std = torch.sqrt(torch.clamp(z_student_sq_sum / max(valid_steps, 1) - z_student_mean.square(), min=0.0)) if latent_dim > 0 else torch.empty(0)
    z_teacher_std = torch.sqrt(torch.clamp(z_teacher_sq_sum / max(valid_steps, 1) - z_teacher_mean.square(), min=0.0)) if latent_dim > 0 else torch.empty(0)
    return (
        float(per_dim.mean().item()),
        per_dim.cpu(),
        float(latent_mse),
        z_student_mean.cpu(),
        z_student_std.cpu(),
        z_teacher_mean.cpu(),
        z_teacher_std.cpu(),
    )


def save_checkpoint(path, model, optimizer, epoch, obs_mean, obs_std, args, metadata, episode_ids, metrics):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "hidden_size": model.hidden_size,
            "mlp_size": model.mlp_size,
            "latent_dim": model.latent_dim,
            "training_episode_ids": [int(value) for value in episode_ids],
            "dataset_metadata": metadata,
            "training_args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def main():
    args = parse_args()
    random.seed(args.selection_seed)
    torch.manual_seed(args.selection_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.selection_seed)

    dataset_dirs = [Path(path).expanduser().resolve() for path in (args.datasets or [args.dataset])]
    output_dir = Path(args.output or os.path.join("logs", "student", datetime.now().strftime("%Y-%m-%d_%H-%M-%S_gru"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes, metadata = load_datasets(dataset_dirs, args.action_clip)
    random.Random(args.selection_seed).shuffle(episodes)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise RuntimeError("No episodes selected for training.")

    val_count = int(round(len(episodes) * args.val_fraction))
    if args.val_fraction > 0.0:
        val_count = max(1, val_count)
    if val_count >= len(episodes):
        raise RuntimeError("Validation split leaves no training episodes.")
    val_episodes = episodes[:val_count]
    train_episodes = episodes[val_count:]
    obs_mean, obs_std = compute_obs_stats(train_episodes)
    device = torch.device(args.device)

    teacher_encoder = None
    latent_stats = None
    latent_mean = None
    latent_std = None
    if args.latent_loss_weight > 0.0:
        teacher_encoder = TeacherPrivilegedEncoder(Path(args.teacher_checkpoint).expanduser().resolve()).to(device)
        latent_stats = compute_latent_stats(train_episodes, teacher_encoder)
        print(
            "[LATENT] teacher_z "
            f"dim={teacher_encoder.output_dim} mean={latent_stats['global_mean']:.6e} "
            f"std={latent_stats['global_std']:.6e} min={latent_stats['global_min']:.6e} "
            f"max={latent_stats['global_max']:.6e} normalize={args.normalize_teacher_latent}"
        )
        if args.normalize_teacher_latent:
            latent_mean = latent_stats["mean"]
            latent_std = latent_stats["std"]

    train_dataset = EpisodeDataset(train_episodes, obs_mean, obs_std, teacher_encoder, latent_mean, latent_std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_episodes,
    )
    eval_episodes = val_episodes if val_episodes else train_episodes
    eval_loader = DataLoader(
        EpisodeDataset(eval_episodes, obs_mean, obs_std, teacher_encoder, latent_mean, latent_std),
        batch_size=min(args.batch_size, len(eval_episodes)),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_episodes,
    )

    obs_dim = train_episodes[0].obs.shape[-1]
    action_dim = train_episodes[0].action.shape[-1]
    if obs_dim != int(metadata.get("student_observation_dim", obs_dim)):
        raise RuntimeError(f"Observation dim {obs_dim} disagrees with metadata {metadata.get('student_observation_dim')}")
    latent_dim = int(teacher_encoder.output_dim) if teacher_encoder is not None else 0
    model = StudentGRU(obs_dim, action_dim, args.hidden_size, args.mlp_size, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    all_train_actions = torch.cat([episode.action for episode in train_episodes], dim=0)
    mean_action = all_train_actions.mean(dim=0)
    baseline_mse = float((all_train_actions - mean_action).square().mean().item())
    print(
        f"[TRAIN] device={device} train_episodes={len(train_episodes)} val_episodes={len(val_episodes)} "
        f"transitions={sum(ep.obs.shape[0] for ep in train_episodes)} baseline_mean_action_mse={baseline_mse:.6e} "
        f"latent_loss_weight={args.latent_loss_weight:.3g}"
    )

    history = []
    best_eval = math.inf
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_squared = torch.zeros(action_dim, device=device)
        epoch_latent_loss = 0.0
        epoch_steps = 0
        for obs, actions, latents, mask, _lengths, _ids in train_loader:
            obs, actions, latents, mask = obs.to(device), actions.to(device), latents.to(device), mask.to(device)
            prediction, z_student, _ = model(obs)
            action_loss, per_dim = masked_action_mse(prediction, actions, mask)
            latent_loss = (
                masked_latent_mse(z_student, latents, mask)
                if args.latent_loss_weight > 0.0 and z_student is not None
                else torch.zeros((), device=device)
            )
            loss = action_loss + float(args.latent_loss_weight) * latent_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            valid = int(mask.sum().item())
            epoch_squared += per_dim.detach() * valid
            epoch_latent_loss += float(latent_loss.detach().item()) * valid
            epoch_steps += valid

        train_per_dim = (epoch_squared / max(epoch_steps, 1)).cpu()
        train_mse = float(train_per_dim.mean().item())
        train_latent_mse = epoch_latent_loss / max(epoch_steps, 1)
        (
            eval_mse,
            eval_per_dim,
            eval_latent_mse,
            z_student_mean,
            z_student_std,
            z_teacher_mean,
            z_teacher_std,
        ) = evaluate(model, eval_loader, device)
        record = {
            "epoch": epoch,
            "train_mse": train_mse,
            "eval_mse": eval_mse,
            "train_latent_mse": train_latent_mse,
            "eval_latent_mse": eval_latent_mse,
            "latent_loss_weight": float(args.latent_loss_weight),
            "z_student_mean": z_student_mean.tolist(),
            "z_student_std": z_student_std.tolist(),
            "z_teacher_mean": z_teacher_mean.tolist(),
            "z_teacher_std": z_teacher_std.tolist(),
        }
        history.append(record)

        if eval_mse < best_eval:
            best_eval = eval_mse
            best_epoch = epoch
            save_checkpoint(
                output_dir / "student_best.pt", model, optimizer, epoch, obs_mean, obs_std, args, metadata,
                [episode.episode_id for episode in train_episodes],
                {
                    "train_mse": train_mse,
                    "eval_mse": eval_mse,
                    "train_latent_mse": train_latent_mse,
                    "eval_latent_mse": eval_latent_mse,
                    "baseline_mse": baseline_mse,
                    "latent_loss_weight": float(args.latent_loss_weight),
                    "normalize_teacher_latent": bool(args.normalize_teacher_latent),
                    "teacher_latent_stats": {
                        key: (value.tolist() if torch.is_tensor(value) else value)
                        for key, value in (latent_stats or {}).items()
                    },
                },
            )

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            per_dim_text = ", ".join(f"a{i}={value:.2e}" for i, value in enumerate(eval_per_dim.tolist()))
            latent_text = ""
            if args.latent_loss_weight > 0.0:
                latent_text = (
                    f" latent_train={train_latent_mse:.6e} latent_eval={eval_latent_mse:.6e} "
                    f"z_s_mean={float(z_student_mean.mean().item()):.3e} z_s_std={float(z_student_std.mean().item()):.3e} "
                    f"z_t_mean={float(z_teacher_mean.mean().item()):.3e} z_t_std={float(z_teacher_std.mean().item()):.3e}"
                )
            print(
                f"[EPOCH {epoch:04d}] train_mse={train_mse:.6e} eval_mse={eval_mse:.6e} "
                f"best={best_eval:.6e}@{best_epoch}{latent_text} | {per_dim_text}"
            )

    save_checkpoint(
        output_dir / "student_last.pt", model, optimizer, args.epochs, obs_mean, obs_std, args, metadata,
        [episode.episode_id for episode in train_episodes],
        {
            "train_mse": history[-1]["train_mse"],
            "eval_mse": history[-1]["eval_mse"],
            "train_latent_mse": history[-1]["train_latent_mse"],
            "eval_latent_mse": history[-1]["eval_latent_mse"],
            "baseline_mse": baseline_mse,
            "latent_loss_weight": float(args.latent_loss_weight),
            "normalize_teacher_latent": bool(args.normalize_teacher_latent),
        },
    )
    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": str(dataset_dirs[0]) if len(dataset_dirs) == 1 else None,
                "datasets": [str(path) for path in dataset_dirs],
                "train_episode_ids": [episode.episode_id for episode in train_episodes],
                "validation_episode_ids": [episode.episode_id for episode in val_episodes],
                "baseline_mean_action_mse": baseline_mse,
                "best_eval_mse": best_eval,
                "best_epoch": best_epoch,
                "latent_loss_weight": float(args.latent_loss_weight),
                "normalize_teacher_latent": bool(args.normalize_teacher_latent),
            },
            file,
            indent=2,
        )
    print(f"[DONE] best_eval_mse={best_eval:.6e} epoch={best_epoch} output={output_dir}")


if __name__ == "__main__":
    main()
