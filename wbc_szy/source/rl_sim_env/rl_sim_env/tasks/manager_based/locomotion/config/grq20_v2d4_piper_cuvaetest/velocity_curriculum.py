"""cuVAETest-local velocity command and performance-driven curricula.

Nothing in this module is exported through the shared locomotion MDP package.
Other tasks therefore retain the legacy combined XY metric and time-driven
velocity curricula.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from isaaclab.utils import configclass

from rl_sim_env.tasks.manager_based.common.mdp.commands import (
    UniformVelocityCommandTerrain,
)
from rl_sim_env.tasks.manager_based.common.mdp.commands_cfg import (
    UniformVelocityCommandTerrainCfg,
)

from .velocity_curriculum_helpers import (
    dwell_complete,
    level_index,
    range_at_level,
    threshold_at_level,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


@configclass
class PerformanceVelocityCommandTerrainCfg(UniformVelocityCommandTerrainCfg):
    """Configuration for cuVAETest's independently advancing x/yaw curriculum."""

    class_type: type = None

    # Explicit ranges make level zero a real stationary stage.  ``max_*_level``
    # remains on the inherited config for checkpoint/force-curriculum compatibility.
    lin_x_level_ranges: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    ang_z_level_ranges: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    lin_y_active_range: tuple[float, float] = (-0.5, 0.5)

    # A threshold is indexed by the level being evaluated.  The final entry is
    # retained for logging/validation even though the final level cannot advance.
    lin_x_mae_thresholds: tuple[float, ...] = (0.08,)
    lin_y_mae_thresholds: tuple[float, ...] = (0.08,)
    ang_z_mae_thresholds: tuple[float, ...] = (0.08,)

    # 200 PPO updates * 24 control steps/update.  This is deliberately expressed
    # in environment control steps so the curriculum does not depend on a runner.
    velocity_curriculum_min_level_steps: int = 4800
    velocity_curriculum_ema_alpha: float = 0.20
    velocity_curriculum_min_survival_rate: float = 0.90
    velocity_curriculum_min_episode_length_ratio: float = 0.90
    velocity_curriculum_min_completed_episodes: int = 128
    velocity_curriculum_required_successes: int = 3
    velocity_curriculum_evaluation_interval_steps: int = 24

    def __post_init__(self) -> None:
        # The implementation is declared below this config class, so bind it at
        # instance construction time (after the module has finished importing).
        self.class_type = PerformanceVelocityCommandTerrain


class PerformanceVelocityCommandTerrain(UniformVelocityCommandTerrain):
    """Terrain command with direct per-episode x/y/yaw tracking MAEs.

    The running means and their explicit sample count live in ``metrics``.  This
    is important because ``LocomotionRLEnv`` clears metric tensors directly on
    episode reset rather than invoking ``CommandTerm.reset``.
    """

    cfg: PerformanceVelocityCommandTerrainCfg

    _STATE_VERSION = 2
    _DISTRIBUTED_STATE_SIZE = 22

    def __init__(self, cfg: PerformanceVelocityCommandTerrainCfg, env: ManagerBasedEnv):
        self._validate_cfg(cfg)
        super().__init__(cfg, env)

        # Replace the shared command's 500-step-normalized combined metrics with
        # direct online episode means.  Zero command samples are valid: level 0
        # explicitly evaluates the robot's ability to remain stationary.
        self.metrics.pop("error_vel_xy", None)
        self.metrics["error_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["valid_vel_samples"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracked_vel_steps"] = torch.zeros(self.num_envs, device=self.device)
        # Bookkeeping for complete/finite episodes.  These are needed by the
        # curriculum statistics, but are not useful training curves.
        self._hidden_log_metrics = frozenset(
            {"valid_vel_samples", "tracked_vel_steps"}
        )

        # Device scalar state avoids a CPU/GPU synchronization for every reset.
        nan = float("nan")
        self._x_mae_ema = torch.tensor(nan, device=self.device)
        self._y_mae_ema = torch.tensor(nan, device=self.device)
        self._yaw_mae_ema = torch.tensor(nan, device=self.device)
        self._survival_rate_ema = torch.tensor(nan, device=self.device)
        self._episode_length_ratio_ema = torch.tensor(nan, device=self.device)
        # Effective episode masses prevent a tiny asynchronous reset batch from
        # receiving the same EMA weight as a batch containing hundreds of
        # completed episodes.  They saturate at the configured reference size.
        self._x_mae_mass = torch.zeros((), device=self.device)
        self._y_mae_mass = torch.zeros((), device=self.device)
        self._yaw_mae_mass = torch.zeros((), device=self.device)
        self._survival_rate_mass = torch.zeros((), device=self.device)
        self._episode_length_ratio_mass = torch.zeros((), device=self.device)
        self._x_episodes_since_level = torch.zeros((), dtype=torch.long, device=self.device)
        self._yaw_episodes_since_level = torch.zeros((), dtype=torch.long, device=self.device)

        initial_step = int(getattr(env, "common_step_counter", 0))
        self._x_level_enter_step = initial_step
        self._yaw_level_enter_step = initial_step
        self._x_successes = 0
        self._yaw_successes = 0
        self._last_statistics_step = -1
        self._last_statistics_had_completed_episodes = torch.zeros(
            (), dtype=torch.bool, device=self.device
        )
        self._last_x_evaluation_step = initial_step
        self._last_yaw_evaluation_step = initial_step

        self.apply_curriculum_ranges()

    @staticmethod
    def _validate_cfg(cfg: PerformanceVelocityCommandTerrainCfg) -> None:
        x_levels = len(cfg.lin_x_level_ranges)
        yaw_levels = len(cfg.ang_z_level_ranges)
        if x_levels != int(round(float(cfg.max_lin_x_level))) + 1:
            raise ValueError(
                "lin_x_level_ranges must contain max_lin_x_level + 1 entries; "
                f"got {x_levels} entries for max level {cfg.max_lin_x_level}."
            )
        if yaw_levels != int(round(float(cfg.max_ang_z_level))) + 1:
            raise ValueError(
                "ang_z_level_ranges must contain max_ang_z_level + 1 entries; "
                f"got {yaw_levels} entries for max level {cfg.max_ang_z_level}."
            )
        if len(cfg.lin_x_mae_thresholds) != x_levels:
            raise ValueError("lin_x_mae_thresholds must match lin_x_level_ranges.")
        if len(cfg.lin_y_mae_thresholds) != x_levels:
            raise ValueError("lin_y_mae_thresholds must match lin_x_level_ranges.")
        if len(cfg.ang_z_mae_thresholds) != yaw_levels:
            raise ValueError("ang_z_mae_thresholds must match ang_z_level_ranges.")
        if not 0.0 < float(cfg.velocity_curriculum_ema_alpha) <= 1.0:
            raise ValueError("velocity_curriculum_ema_alpha must be in (0, 1].")
        if int(cfg.velocity_curriculum_min_level_steps) < 0:
            raise ValueError("velocity_curriculum_min_level_steps must be non-negative.")
        if int(cfg.velocity_curriculum_min_completed_episodes) < 1:
            raise ValueError("velocity_curriculum_min_completed_episodes must be positive.")
        if int(cfg.velocity_curriculum_required_successes) < 1:
            raise ValueError("velocity_curriculum_required_successes must be positive.")
        if int(cfg.velocity_curriculum_evaluation_interval_steps) < 1:
            raise ValueError("velocity_curriculum_evaluation_interval_steps must be positive.")
        range_at_level(cfg.lin_x_level_ranges, cfg.lin_x_level)
        range_at_level(cfg.ang_z_level_ranges, cfg.ang_z_level)
        for level in range(x_levels):
            threshold_at_level(cfg.lin_x_mae_thresholds, level)
            threshold_at_level(cfg.lin_y_mae_thresholds, level)
        for level in range(yaw_levels):
            threshold_at_level(cfg.ang_z_mae_thresholds, level)

    def _update_metrics(self) -> None:
        """Update direct episode MAEs with an explicit per-environment count."""
        self._accumulate_metric_samples(
            slice(None), self._env.episode_length_buf > 0
        )

    def _accumulate_metric_samples(
        self, env_ids: Sequence[int] | slice, tracking_valid: torch.Tensor
    ) -> None:
        """Accumulate selected tracking samples into per-environment means."""
        errors = torch.stack(
            (
                torch.abs(
                    self.vel_command_b[env_ids, 0]
                    - self.robot.data.root_lin_vel_b[env_ids, 0]
                ),
                torch.abs(
                    self.vel_command_b[env_ids, 1]
                    - self.robot.data.root_lin_vel_b[env_ids, 1]
                ),
                torch.abs(
                    self.vel_command_b[env_ids, 2]
                    - self.robot.data.root_ang_vel_b[env_ids, 2]
                ),
            ),
            dim=-1,
        )
        tracked_steps = self.metrics["tracked_vel_steps"][env_ids]
        self.metrics["tracked_vel_steps"][env_ids] = tracked_steps + tracking_valid.to(
            dtype=tracked_steps.dtype
        )
        valid = torch.isfinite(errors).all(dim=-1) & tracking_valid
        old_count = self.metrics["valid_vel_samples"][env_ids]
        new_count = old_count + valid.to(old_count.dtype)
        safe_count = new_count.clamp_min(1.0)

        for index, name in enumerate(("error_vel_x", "error_vel_y", "error_vel_yaw")):
            mean = self.metrics[name][env_ids]
            updated = mean + (errors[:, index] - mean) / safe_count
            self.metrics[name][env_ids] = torch.where(valid, updated, mean)
        self.metrics["valid_vel_samples"][env_ids] = new_count

    @staticmethod
    def _episode_weighted_ema(
        previous: torch.Tensor,
        episode_mass: torch.Tensor,
        sample: torch.Tensor,
        sample_count: torch.Tensor,
        *,
        alpha: float,
        reference_episodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update an EMA with weight proportional to completed episodes.

        Before ``reference_episodes`` have contributed, this is an exact
        episode-weighted warm-up mean.  Afterwards, ``alpha`` is interpreted as
        the weight of one full reference-sized batch; small asynchronous reset
        batches receive only their proportional fraction of that weight.
        """
        reference = float(max(int(reference_episodes), 1))
        count = sample_count.to(dtype=previous.dtype)
        valid = torch.isfinite(sample) & (count > 0.0)
        safe_count = count.clamp_min(1.0)
        combined_mass = episode_mass + count

        previous_sum = torch.where(
            torch.isfinite(previous), previous * episode_mass, torch.zeros_like(previous)
        )
        warmup = (previous_sum + sample * count) / combined_mass.clamp_min(1.0)

        effective_alpha = 1.0 - (1.0 - float(alpha)) ** (safe_count / reference)
        steady = previous + effective_alpha * (sample - previous)
        use_warmup = (~torch.isfinite(previous)) | (episode_mass < reference)
        updated = torch.where(use_warmup, warmup, steady)
        updated = torch.where(valid, updated, previous)
        updated_mass = torch.where(
            valid,
            combined_mass.clamp_max(reference),
            episode_mass,
        )
        return updated, updated_mass

    def ingest_completed_episodes(
        self, env: ManagerBasedRLEnv, env_ids: Sequence[int]
    ) -> torch.Tensor:
        """Fold one reset batch into robust command-owned curriculum statistics."""
        # During the environment's very first/manual reset there is no preceding
        # termination snapshot yet.  Such a reset contains no completed episode
        # and must not be treated as either a failure or a timeout.
        reset_time_outs = getattr(env, "reset_time_outs", None)
        reset_buf = getattr(env, "reset_buf", None)
        if reset_time_outs is None or reset_buf is None:
            self._last_statistics_had_completed_episodes.zero_()
            return self._last_statistics_had_completed_episodes

        step = int(env.common_step_counter)
        # x and yaw terms run back-to-back for the same reset batch.
        if step == self._last_statistics_step:
            return self._last_statistics_had_completed_episodes
        self._last_statistics_step = step

        episode_lengths = env.episode_length_buf[env_ids].float()
        # Command metrics are normally computed after reset, so explicitly fold
        # the still-available terminal state into the old episode before the
        # scene and metrics are cleared.
        terminal_tracking_valid = (
            reset_buf[env_ids].bool() & (episode_lengths > 0.0)
        )
        self._accumulate_metric_samples(env_ids, terminal_tracking_valid)

        counts = self.metrics["valid_vel_samples"][env_ids]
        tracked_steps = self.metrics["tracked_vel_steps"][env_ids]
        # ``learn(init_at_random_ep_len=True)`` advances only the logical length
        # of the first episode.  The actual tracking-step count exposes that bootstrap
        # fragment: a genuine episode now has one observed sample per control
        # step, including the terminal state folded above.
        minimum_full_episode_samples = episode_lengths
        complete_episode = (
            (episode_lengths > 0.0)
            & (tracked_steps >= minimum_full_episode_samples)
            & reset_buf[env_ids].bool()
        )
        # A genuine one-step failure has no non-terminal velocity sample, but it
        # must still lower the survival/length statistics.  Error MAEs require
        # at least one observed sample and are gated independently.
        valid_error = complete_episode & (counts > 0.0) & (counts >= tracked_steps)
        for name in ("error_vel_x", "error_vel_y", "error_vel_yaw"):
            valid_error &= torch.isfinite(self.metrics[name][env_ids])

        completed_episodes = complete_episode.sum()
        completed_errors = valid_error.sum()
        self._last_statistics_had_completed_episodes.copy_(completed_episodes > 0)
        safe_completed_errors = completed_errors.clamp_min(1).to(dtype=counts.dtype)

        def batch_mean(name: str) -> torch.Tensor:
            values = self.metrics[name][env_ids]
            return (
                torch.where(valid_error, values, torch.zeros_like(values)).sum()
                / safe_completed_errors
            )

        alpha = float(self.cfg.velocity_curriculum_ema_alpha)
        reference_episodes = int(self.cfg.velocity_curriculum_min_completed_episodes)
        # This method runs inside the runner's torch.inference_mode() rollout.
        # Keep the persistent, construction-time tensors and copy results into
        # them; rebinding an attribute to ``updated_*`` would make checkpoint/DDP
        # restoration mutate an inference tensor outside inference mode.
        updated_ema, updated_mass = self._episode_weighted_ema(
            self._x_mae_ema,
            self._x_mae_mass,
            batch_mean("error_vel_x"),
            completed_errors,
            alpha=alpha,
            reference_episodes=reference_episodes,
        )
        self._x_mae_ema.copy_(updated_ema)
        self._x_mae_mass.copy_(updated_mass)

        updated_ema, updated_mass = self._episode_weighted_ema(
            self._y_mae_ema,
            self._y_mae_mass,
            batch_mean("error_vel_y"),
            completed_errors,
            alpha=alpha,
            reference_episodes=reference_episodes,
        )
        self._y_mae_ema.copy_(updated_ema)
        self._y_mae_mass.copy_(updated_mass)

        updated_ema, updated_mass = self._episode_weighted_ema(
            self._yaw_mae_ema,
            self._yaw_mae_mass,
            batch_mean("error_vel_yaw"),
            completed_errors,
            alpha=alpha,
            reference_episodes=reference_episodes,
        )
        self._yaw_mae_ema.copy_(updated_ema)
        self._yaw_mae_mass.copy_(updated_mass)

        self._x_episodes_since_level += completed_episodes
        self._yaw_episodes_since_level += completed_episodes

        length_ratio = episode_lengths / max(float(env.max_episode_length), 1.0)
        safe_completed_episodes = completed_episodes.clamp_min(1).to(dtype=counts.dtype)
        batch_length_ratio = (
            torch.where(complete_episode, length_ratio, torch.zeros_like(length_ratio)).sum()
            / safe_completed_episodes
        )
        time_outs = reset_time_outs[env_ids].float()
        batch_survival_rate = (
            torch.where(complete_episode, time_outs, torch.zeros_like(time_outs)).sum()
            / safe_completed_episodes
        )
        updated_ema, updated_mass = self._episode_weighted_ema(
            self._episode_length_ratio_ema,
            self._episode_length_ratio_mass,
            batch_length_ratio,
            completed_episodes,
            alpha=alpha,
            reference_episodes=reference_episodes,
        )
        self._episode_length_ratio_ema.copy_(updated_ema)
        self._episode_length_ratio_mass.copy_(updated_mass)

        updated_ema, updated_mass = self._episode_weighted_ema(
            self._survival_rate_ema,
            self._survival_rate_mass,
            batch_survival_rate,
            completed_episodes,
            alpha=alpha,
            reference_episodes=reference_episodes,
        )
        self._survival_rate_ema.copy_(updated_ema)
        self._survival_rate_mass.copy_(updated_mass)
        return self._last_statistics_had_completed_episodes

    def _episode_length_gate(self) -> torch.Tensor:
        """Require an average completed episode length strictly above 900 steps."""
        average_episode_length = (
            self._episode_length_ratio_ema * float(self._env.max_episode_length)
        )
        minimum_episode_length = (
            float(self.cfg.velocity_curriculum_min_episode_length_ratio)
            * float(self._env.max_episode_length)
        )
        return (
            torch.isfinite(average_episode_length)
            & (average_episode_length > minimum_episode_length)
        )

    @staticmethod
    def _is_curriculum_authority() -> bool:
        """Only rank zero decides levels; the runner broadcasts its state."""
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        return world_size <= 1 or rank == 0

    def maybe_advance_lin_x(
        self, current_step: int, has_completed_episodes: torch.Tensor
    ) -> bool:
        """Advance only x (and activate y at x level 1) when all gates pass."""
        if not self._is_curriculum_authority():
            return False
        level = level_index(self.cfg.lin_x_level, len(self.cfg.lin_x_level_ranges))
        if level >= int(round(float(self.cfg.max_lin_x_level))):
            return False
        interval = int(self.cfg.velocity_curriculum_evaluation_interval_steps)
        if int(current_step) - self._last_x_evaluation_step < interval:
            return False
        if not bool(has_completed_episodes.item()):
            return False
        self._last_x_evaluation_step = int(current_step)

        dwell_ok = dwell_complete(
            current_step,
            self._x_level_enter_step,
            int(self.cfg.velocity_curriculum_min_level_steps),
        )
        gate = (
            self._episode_length_gate()
            & torch.isfinite(self._x_mae_ema)
            & torch.isfinite(self._y_mae_ema)
            & (self._x_mae_ema <= threshold_at_level(self.cfg.lin_x_mae_thresholds, level))
            & (self._y_mae_ema <= threshold_at_level(self.cfg.lin_y_mae_thresholds, level))
        )
        passed = dwell_ok and bool(gate.item())
        self._x_successes = self._x_successes + 1 if passed else 0
        if self._x_successes < int(self.cfg.velocity_curriculum_required_successes):
            return False

        self.cfg.lin_x_level = float(level + 1)
        self._x_level_enter_step = int(current_step)
        self._x_successes = 0
        self._x_episodes_since_level.zero_()
        self._x_mae_ema.fill_(float("nan"))
        self._y_mae_ema.fill_(float("nan"))
        self._x_mae_mass.zero_()
        self._y_mae_mass.zero_()
        self.apply_curriculum_ranges()
        return True

    def maybe_advance_ang_z(
        self, current_step: int, has_completed_episodes: torch.Tensor
    ) -> bool:
        """Advance yaw independently of the x/y curriculum."""
        if not self._is_curriculum_authority():
            return False
        level = level_index(self.cfg.ang_z_level, len(self.cfg.ang_z_level_ranges))
        if level >= int(round(float(self.cfg.max_ang_z_level))):
            return False
        interval = int(self.cfg.velocity_curriculum_evaluation_interval_steps)
        if int(current_step) - self._last_yaw_evaluation_step < interval:
            return False
        if not bool(has_completed_episodes.item()):
            return False
        self._last_yaw_evaluation_step = int(current_step)

        dwell_ok = dwell_complete(
            current_step,
            self._yaw_level_enter_step,
            int(self.cfg.velocity_curriculum_min_level_steps),
        )
        gate = (
            self._episode_length_gate()
            & torch.isfinite(self._yaw_mae_ema)
            & (
                self._yaw_mae_ema
                <= threshold_at_level(self.cfg.ang_z_mae_thresholds, level)
            )
        )
        passed = dwell_ok and bool(gate.item())
        self._yaw_successes = self._yaw_successes + 1 if passed else 0
        if self._yaw_successes < int(self.cfg.velocity_curriculum_required_successes):
            return False

        self.cfg.ang_z_level = float(level + 1)
        self._yaw_level_enter_step = int(current_step)
        self._yaw_successes = 0
        self._yaw_episodes_since_level.zero_()
        self._yaw_mae_ema.fill_(float("nan"))
        self._yaw_mae_mass.zero_()
        self.apply_curriculum_ranges()
        return True

    def apply_curriculum_ranges(self) -> None:
        """Materialize current x/y/yaw levels into every task-local terrain range."""
        x_level = level_index(self.cfg.lin_x_level, len(self.cfg.lin_x_level_ranges))
        x_range = range_at_level(self.cfg.lin_x_level_ranges, x_level)
        yaw_range = range_at_level(self.cfg.ang_z_level_ranges, self.cfg.ang_z_level)
        y_range = (0.0, 0.0) if x_level == 0 else tuple(self.cfg.lin_y_active_range)

        for range_cfg in self.cfg.ranges.values():
            range_cfg.lin_vel_x = x_range
            range_cfg.lin_vel_y = y_range
            range_cfg.ang_vel_z = yaw_range

        # ManagerBase owns a deepcopy of the environment config.  Keep the
        # original env config synchronized as well so logs/exported configs do
        # not misleadingly remain at level zero while runtime has advanced.
        env_command_cfg = getattr(
            getattr(getattr(self._env, "cfg", None), "commands", None),
            "base_command",
            None,
        )
        if env_command_cfg is not None and env_command_cfg is not self.cfg:
            env_command_cfg.lin_x_level = float(self.cfg.lin_x_level)
            env_command_cfg.ang_z_level = float(self.cfg.ang_z_level)
            for key, range_cfg in self.cfg.ranges.items():
                if key not in env_command_cfg.ranges:
                    continue
                env_range_cfg = env_command_cfg.ranges[key]
                env_range_cfg.lin_vel_x = tuple(range_cfg.lin_vel_x)
                env_range_cfg.lin_vel_y = tuple(range_cfg.lin_vel_y)
                env_range_cfg.ang_vel_z = tuple(range_cfg.ang_vel_z)

        # Heading commands clip through these per-env buffers.  Keep them in sync
        # immediately after level changes and checkpoint restoration.
        if hasattr(self, "ang_vel_z_limit_low"):
            for key, ids in self.cfg.command_ids.items():
                self.ang_vel_z_limit_low[ids] = self.cfg.ranges[key].ang_vel_z[0]
                self.ang_vel_z_limit_high[ids] = self.cfg.ranges[key].ang_vel_z[1]

    def distributed_curriculum_state_tensor(self) -> torch.Tensor:
        """Pack all restorable curriculum state into one fixed-layout CUDA tensor."""
        scalar_state = torch.tensor(
            (
                float(self._STATE_VERSION),
                float(self.cfg.lin_x_level),
                float(self.cfg.ang_z_level),
                float(self._x_level_enter_step),
                float(self._yaw_level_enter_step),
                float(self._x_successes),
                float(self._yaw_successes),
                float(self._last_statistics_step),
                float(self._last_x_evaluation_step),
                float(self._last_yaw_evaluation_step),
            ),
            dtype=torch.float64,
            device=self.device,
        )
        statistic_state = torch.stack(
            (
                self._x_mae_ema,
                self._y_mae_ema,
                self._yaw_mae_ema,
                self._survival_rate_ema,
                self._episode_length_ratio_ema,
                self._x_mae_mass,
                self._y_mae_mass,
                self._yaw_mae_mass,
                self._survival_rate_mass,
                self._episode_length_ratio_mass,
            )
        ).to(dtype=torch.float64)
        episode_state = torch.stack(
            (self._x_episodes_since_level, self._yaw_episodes_since_level)
        ).to(dtype=torch.float64)
        return torch.cat((scalar_state, statistic_state, episode_state))

    @classmethod
    def _curriculum_state_from_flat_values(
        cls, values: Sequence[float]
    ) -> dict[str, Any]:
        """Decode the fixed DDP layout into the checkpoint-compatible schema."""
        if len(values) != cls._DISTRIBUTED_STATE_SIZE:
            raise ValueError(
                "Distributed curriculum state must contain "
                f"{cls._DISTRIBUTED_STATE_SIZE} values, got {len(values)}."
            )

        def optional_float(index: int) -> float | None:
            value = float(values[index])
            return value if value == value else None

        return {
            "version": int(values[0]),
            "lin_x_level": float(values[1]),
            "ang_z_level": float(values[2]),
            "x_level_enter_step": int(values[3]),
            "yaw_level_enter_step": int(values[4]),
            "x_successes": int(values[5]),
            "yaw_successes": int(values[6]),
            "last_statistics_step": int(values[7]),
            "last_x_evaluation_step": int(values[8]),
            "last_yaw_evaluation_step": int(values[9]),
            "x_mae_ema": optional_float(10),
            "y_mae_ema": optional_float(11),
            "yaw_mae_ema": optional_float(12),
            "survival_rate_ema": optional_float(13),
            "episode_length_ratio_ema": optional_float(14),
            "x_mae_mass": optional_float(15),
            "y_mae_mass": optional_float(16),
            "yaw_mae_mass": optional_float(17),
            "survival_rate_mass": optional_float(18),
            "episode_length_ratio_mass": optional_float(19),
            "x_episodes_since_level": int(values[20]),
            "yaw_episodes_since_level": int(values[21]),
        }

    def load_distributed_curriculum_state_tensor(self, state: torch.Tensor) -> None:
        """Restore rank-zero's complete state from one device-to-host transfer."""
        if not isinstance(state, torch.Tensor):
            raise TypeError(
                f"Distributed curriculum state must be a tensor, got {type(state).__name__}."
            )
        if state.shape != (self._DISTRIBUTED_STATE_SIZE,):
            raise ValueError(
                "Distributed curriculum state must have shape "
                f"({self._DISTRIBUTED_STATE_SIZE},), got {tuple(state.shape)}."
            )
        values = state.detach().cpu().tolist()
        decoded = self._curriculum_state_from_flat_values(values)
        levels_changed = (
            float(decoded["lin_x_level"]) != float(self.cfg.lin_x_level)
            or float(decoded["ang_z_level"]) != float(self.cfg.ang_z_level)
        )
        self.load_curriculum_state_dict(decoded, apply_ranges=levels_changed)

    def curriculum_state_dict(self) -> dict[str, Any]:
        """Return checkpoint state with a single batched device-to-host transfer."""
        values = self.distributed_curriculum_state_tensor().detach().cpu().tolist()
        return self._curriculum_state_from_flat_values(values)

    def load_curriculum_state_dict(
        self, state: dict[str, Any], *, apply_ranges: bool = True
    ) -> None:
        """Restore new state or initialize it safely from a legacy level-only state."""
        if not isinstance(state, dict):
            raise TypeError(f"Curriculum state must be a dict, got {type(state).__name__}.")
        version = state.get("version")
        if version is not None and int(version) not in (1, self._STATE_VERSION):
            raise ValueError(f"Unsupported velocity curriculum state version: {version}.")
        legacy_levels = bool(state.get("legacy_time_driven_levels", False))
        if "lin_x_level" in state:
            level = float(state["lin_x_level"]) + (1.0 if legacy_levels else 0.0)
            self.cfg.lin_x_level = min(level, float(self.cfg.max_lin_x_level))
        if "ang_z_level" in state:
            level = float(state["ang_z_level"]) + (1.0 if legacy_levels else 0.0)
            self.cfg.ang_z_level = min(level, float(self.cfg.max_ang_z_level))
        # Validate restored levels before mutating any command ranges.
        level_index(self.cfg.lin_x_level, len(self.cfg.lin_x_level_ranges))
        level_index(self.cfg.ang_z_level, len(self.cfg.ang_z_level_ranges))

        fallback_step = int(state.get("common_step_counter", 0))
        self._x_level_enter_step = int(state.get("x_level_enter_step", fallback_step))
        self._yaw_level_enter_step = int(state.get("yaw_level_enter_step", fallback_step))
        self._x_successes = int(state.get("x_successes", 0))
        self._yaw_successes = int(state.get("yaw_successes", 0))
        self._last_statistics_step = int(state.get("last_statistics_step", fallback_step))
        self._last_x_evaluation_step = int(
            state.get("last_x_evaluation_step", fallback_step)
        )
        self._last_yaw_evaluation_step = int(
            state.get("last_yaw_evaluation_step", fallback_step)
        )
        # A reset immediately after loading is synthetic, not another completed
        # episode at the saved step.  The same-step dedup path must return false.
        self._last_statistics_had_completed_episodes.zero_()

        def restore_scalar(target: torch.Tensor, key: str) -> None:
            value = state.get(key)
            target.fill_(float("nan") if value is None else float(value))

        restore_scalar(self._x_mae_ema, "x_mae_ema")
        restore_scalar(self._y_mae_ema, "y_mae_ema")
        restore_scalar(self._yaw_mae_ema, "yaw_mae_ema")
        restore_scalar(self._survival_rate_ema, "survival_rate_ema")
        restore_scalar(self._episode_length_ratio_ema, "episode_length_ratio_ema")
        reference_mass = float(self.cfg.velocity_curriculum_min_completed_episodes)
        for target, key, ema_key in (
            (self._x_mae_mass, "x_mae_mass", "x_mae_ema"),
            (self._y_mae_mass, "y_mae_mass", "y_mae_ema"),
            (self._yaw_mae_mass, "yaw_mae_mass", "yaw_mae_ema"),
            (self._survival_rate_mass, "survival_rate_mass", "survival_rate_ema"),
            (
                self._episode_length_ratio_mass,
                "episode_length_ratio_mass",
                "episode_length_ratio_ema",
            ),
        ):
            # Version 1 stored finite EMAs before episode masses existed.  Treat
            # those restored estimates as fully warmed instead of letting the
            # next tiny reset batch overwrite them.
            if key in state:
                value = state[key]
            else:
                value = reference_mass if state.get(ema_key) is not None else 0.0
            target.fill_(0.0 if value is None else float(value))
        self._x_episodes_since_level.fill_(int(state.get("x_episodes_since_level", 0)))
        self._yaw_episodes_since_level.fill_(int(state.get("yaw_episodes_since_level", 0)))
        if apply_ranges:
            self.apply_curriculum_ranges()

    def curriculum_log_state(self, *, axis: str) -> dict[str, torch.Tensor]:
        """Expose only the visible curriculum result, not its gate internals.

        Per-episode tracking MAEs are already emitted under ``Metrics``.  The
        EMA, sample counters, dwell timer, and thresholds are retained as
        internal state for a stable promotion decision, but would be redundant
        and confusing as dashboard curves.
        """
        if axis == "lin_x":
            level = float(self.cfg.lin_x_level)
        elif axis == "ang_z":
            level = float(self.cfg.ang_z_level)
        else:
            raise ValueError(f"Unsupported curriculum axis: {axis}.")
        return {
            "level": torch.tensor(level, device=self.device),
        }


# Resolve the forward reference after both config and implementation exist.
PerformanceVelocityCommandTerrainCfg.class_type = PerformanceVelocityCommandTerrain


def create_performance_velocity_command_terrain_cfg(
    *,
    command_ids: dict[str, list[int]],
    ranges: dict[str, UniformVelocityCommandTerrainCfg.Ranges],
    lin_x_level: float,
    ang_z_level: float,
    max_lin_x_level: float,
    max_ang_z_level: float,
    vel_curriculum_episode_mult: float,
    heading_control_stiffness: float,
    lin_x_level_ranges: tuple[tuple[float, float], ...],
    ang_z_level_ranges: tuple[tuple[float, float], ...],
    lin_y_active_range: tuple[float, float],
    lin_x_mae_thresholds: tuple[float, ...],
    lin_y_mae_thresholds: tuple[float, ...],
    ang_z_mae_thresholds: tuple[float, ...],
    velocity_curriculum_min_level_steps: int,
    velocity_curriculum_ema_alpha: float,
    velocity_curriculum_min_survival_rate: float,
    velocity_curriculum_min_episode_length_ratio: float,
    velocity_curriculum_min_completed_episodes: int,
    velocity_curriculum_required_successes: int,
    velocity_curriculum_evaluation_interval_steps: int,
) -> PerformanceVelocityCommandTerrainCfg:
    """Construct the task-local command without changing the shared factory."""
    return PerformanceVelocityCommandTerrainCfg(
        class_type=PerformanceVelocityCommandTerrain,
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        command_ids=command_ids,
        ranges=ranges,
        lin_x_level=lin_x_level,
        ang_z_level=ang_z_level,
        max_lin_x_level=max_lin_x_level,
        max_ang_z_level=max_ang_z_level,
        vel_curriculum_episode_mult=vel_curriculum_episode_mult,
        heading_control_stiffness=heading_control_stiffness,
        lin_x_level_ranges=lin_x_level_ranges,
        ang_z_level_ranges=ang_z_level_ranges,
        lin_y_active_range=lin_y_active_range,
        lin_x_mae_thresholds=lin_x_mae_thresholds,
        lin_y_mae_thresholds=lin_y_mae_thresholds,
        ang_z_mae_thresholds=ang_z_mae_thresholds,
        velocity_curriculum_min_level_steps=velocity_curriculum_min_level_steps,
        velocity_curriculum_ema_alpha=velocity_curriculum_ema_alpha,
        velocity_curriculum_min_survival_rate=velocity_curriculum_min_survival_rate,
        velocity_curriculum_min_episode_length_ratio=(
            velocity_curriculum_min_episode_length_ratio
        ),
        velocity_curriculum_min_completed_episodes=velocity_curriculum_min_completed_episodes,
        velocity_curriculum_required_successes=velocity_curriculum_required_successes,
        velocity_curriculum_evaluation_interval_steps=(
            velocity_curriculum_evaluation_interval_steps
        ),
    )


def _performance_command(env: ManagerBasedRLEnv) -> PerformanceVelocityCommandTerrain:
    command = env.command_manager.get_term("base_command")
    if not isinstance(command, PerformanceVelocityCommandTerrain):
        raise TypeError(
            "cuVAETest performance curriculum requires PerformanceVelocityCommandTerrain, "
            f"got {type(command).__name__}."
        )
    return command


def lin_vel_x_command_threshold(
    env: ManagerBasedRLEnv, env_ids: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Task-local x/y performance gate with the legacy curriculum-term signature."""
    command = _performance_command(env)
    has_completed_episodes = command.ingest_completed_episodes(env, env_ids)
    command.maybe_advance_lin_x(
        int(env.common_step_counter), has_completed_episodes
    )
    return command.curriculum_log_state(axis="lin_x")


def ang_vel_z_command_threshold(
    env: ManagerBasedRLEnv, env_ids: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Task-local, independently advancing yaw performance gate."""
    command = _performance_command(env)
    has_completed_episodes = command.ingest_completed_episodes(env, env_ids)
    command.maybe_advance_ang_z(
        int(env.common_step_counter), has_completed_episodes
    )
    return command.curriculum_log_state(axis="ang_z")


__all__ = [
    "PerformanceVelocityCommandTerrain",
    "PerformanceVelocityCommandTerrainCfg",
    "ang_vel_z_command_threshold",
    "create_performance_velocity_command_terrain_cfg",
    "lin_vel_x_command_threshold",
]
