# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict

from torch.utils.tensorboard import SummaryWriter

try:
    import swanlab
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "SwanLab is required to log to SwanLab. Please `pip install swanlab`."
    ) from e


class SwanLabSummaryWriter(SummaryWriter):
    """Summary writer for SwanLab."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        project = cfg.get("swanlab_project", None)
        if project is None:
            raise KeyError(
                "Please specify swanlab_project in the runner config."
            )

        # Initialize swanlab (be tolerant to API differences)
        try:
            swanlab.init(project=project, experiment_name=run_name)
        except TypeError:
            swanlab.init(project=project)

        # Store log directory using config API (supports strings)
        # Don't use log() for strings as it only supports numeric types
        try:
            if hasattr(swanlab, "config") and hasattr(
                swanlab.config, "update"
            ):
                swanlab.config.update({"log_dir": log_dir})
            # If config API doesn't exist, skip logging log_dir
            # as swanlab.log() doesn't support string types
        except Exception:
            pass

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def _flatten_dict(self, d, parent_key="", sep="."):
        """Flatten nested dictionary, only including numeric values."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float, bool)):
                # Only include numeric and boolean values
                items.append((new_key, v))
            # Skip None, strings, tuples, lists, and other types
        return dict(items)

    def store_config(self, env_cfg, runner_cfg, alg_cfg=None, policy_cfg=None):
        """Store configuration to SwanLab, only logging numeric values."""
        try:
            # Try to use swanlab.config if available (for config data)
            if hasattr(swanlab, "config") and hasattr(
                swanlab.config, "update"
            ):
                # Use config API for full configuration (supports all types)
                payload = {"runner_cfg": runner_cfg}
                if alg_cfg is not None:
                    payload["algorithm_cfg"] = alg_cfg
                if policy_cfg is not None:
                    payload["policy_cfg"] = policy_cfg
                try:
                    payload["env_cfg"] = env_cfg.to_dict()
                except Exception:
                    payload["env_cfg"] = asdict(env_cfg)

                try:
                    swanlab.config.update(payload)
                except Exception:
                    pass
            else:
                # Fallback: Only log numeric config values using log()
                # This prevents type errors but only logs numeric configs
                all_configs = {}

                # Process runner_cfg
                if isinstance(runner_cfg, dict):
                    all_configs.update(
                        self._flatten_dict(
                            runner_cfg, parent_key="runner_cfg"
                        )
                    )

                # Process alg_cfg
                if alg_cfg is not None and isinstance(alg_cfg, dict):
                    all_configs.update(
                        self._flatten_dict(
                            alg_cfg, parent_key="algorithm_cfg"
                        )
                    )

                # Process policy_cfg
                if policy_cfg is not None and isinstance(policy_cfg, dict):
                    all_configs.update(
                        self._flatten_dict(
                            policy_cfg, parent_key="policy_cfg"
                        )
                    )

                # Process env_cfg
                try:
                    env_cfg_dict = env_cfg.to_dict()
                except Exception:
                    env_cfg_dict = asdict(env_cfg)

                if isinstance(env_cfg_dict, dict):
                    all_configs.update(
                        self._flatten_dict(
                            env_cfg_dict, parent_key="env_cfg"
                        )
                    )

                # Only log if we have numeric values
                if all_configs:
                    try:
                        swanlab.log(all_configs)
                    except Exception:
                        pass
        except Exception:
            # Silently fail if config logging doesn't work
            pass

    def add_scalar(
        self,
        tag,
        scalar_value,
        global_step=None,
        walltime=None,
        new_style=False,
    ):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        key = self._map_path(tag)
        # SwanLab's log() accepts a dict and optional step parameter
        try:
            if global_step is not None:
                swanlab.log({key: scalar_value}, step=global_step)
            else:
                swanlab.log({key: scalar_value})
        except TypeError:
            # Fallback if step parameter is not supported
            data = {key: scalar_value}
            if global_step is not None:
                data["step"] = global_step
            swanlab.log(data)

    def stop(self):
        try:
            swanlab.finish()
        except Exception:
            pass

    def log_config(self, env_cfg, runner_cfg, alg_cfg=None, policy_cfg=None):
        self.store_config(
            env_cfg, runner_cfg, alg_cfg=alg_cfg, policy_cfg=policy_cfg
        )

    def save_model(self, model_path, iter):
        # SwanLab does not necessarily provide a "save" API like wandb;
        # keep file on disk.
        _ = (model_path, iter)

    def save_file(self, path, iter=None):
        _ = (path, iter)

    def _map_path(self, path: str) -> str:
        return self.name_map.get(path, path)
