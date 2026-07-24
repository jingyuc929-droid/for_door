#!/usr/bin/env python3
"""Export a locomotion checkpoint to ONNX without launching Isaac Sim.

This exporter reconstructs the policy and VAE directly from the training-time
``params/agent.yaml`` and loads their state dictionaries on CPU.  It therefore
does not create a Gym environment and does not reserve GPU memory.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ruamel.yaml import YAML
from tensordict import TensorDict

from rl_algorithms.rsl_rl.modules import ActorCriticEncoder, VAEBlind


class OfflineLocomotionPolicy(torch.nn.Module):
    """Combined deterministic VAE encoder and actor used by deployment."""

    def __init__(self, actor: ActorCriticEncoder, vae: VAEBlind):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu().eval()
        self.vae = copy.deepcopy(vae).cpu().eval()

    def forward(self, actor_obs: torch.Tensor, vae_obs: torch.Tensor) -> torch.Tensor:
        estimator_out = self.vae.act_inference(vae_obs)
        observations = TensorDict(
            {"actor_obs": actor_obs, "estimator_out": estimator_out},
            batch_size=actor_obs.shape[:-1],
        )
        return self.actor.act_inference(observations)


def export_and_verify(actor: ActorCriticEncoder, vae: VAEBlind, onnx_path: Path) -> None:
    policy = OfflineLocomotionPolicy(actor, vae)
    actor_input_dim = next(module.in_features for module in actor.actor.modules() if isinstance(module, torch.nn.Linear))
    vae_input_dim = next(module.in_features for module in vae.encoder.modules() if isinstance(module, torch.nn.Linear))
    with torch.inference_mode():
        estimator_dim = vae.act_inference(torch.zeros(1, vae_input_dim)).shape[-1]
    actor_obs_dim = actor_input_dim - estimator_dim
    if actor_obs_dim <= 0:
        raise ValueError(f"Invalid inferred actor observation dimension: {actor_obs_dim}")

    dummy_actor = torch.zeros(1, actor_obs_dim, dtype=torch.float32)
    dummy_vae = torch.zeros(1, vae_input_dim, dtype=torch.float32)
    torch.onnx.export(
        policy,
        (dummy_actor, dummy_vae),
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["actor_obs", "vae_obs"],
        output_names=["actions"],
        dynamic_axes={
            "actor_obs": {0: "batch_size"},
            "vae_obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        },
    )

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_actions = session.run(
        ["actions"],
        {"actor_obs": dummy_actor.numpy(), "vae_obs": dummy_vae.numpy()},
    )[0]
    with torch.inference_mode():
        torch_actions = policy(dummy_actor, dummy_vae).numpy()
    np.testing.assert_allclose(onnx_actions, torch_actions, rtol=1e-5, atol=1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to model_*.pt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <experiment>/exported/<run>/<checkpoint-stem>)",
    )
    return parser.parse_args()


def default_output_dir(checkpoint: Path) -> Path:
    run_dir = checkpoint.parent
    return run_dir.parent / "exported" / run_dir.name / checkpoint.stem


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    run_dir = checkpoint.parent
    agent_yaml_path = run_dir / "params" / "agent.yaml"
    policy_yaml_path = run_dir / "policy.yaml"
    if not agent_yaml_path.is_file():
        raise FileNotFoundError(f"Training-time agent config does not exist: {agent_yaml_path}")
    if not policy_yaml_path.is_file():
        raise FileNotFoundError(f"Training-time policy config does not exist: {policy_yaml_path}")

    yaml = YAML()
    with agent_yaml_path.open(encoding="utf-8") as stream:
        agent_cfg = yaml.load(stream)

    policy_type = agent_cfg["policy_type"]
    if policy_type["actor_critic_type"] != "ActorCriticEncoder" or policy_type["vae_type"] != "VAEBlind":
        raise ValueError(
            "This offline exporter currently supports ActorCriticEncoder + VAEBlind, got "
            f"{policy_type['actor_critic_type']} + {policy_type['vae_type']}"
        )

    module_cfg = copy.deepcopy(agent_cfg["module_cfg_dict"])
    module_cfg["actor_critic"]["min_normalized_std"] = torch.as_tensor(
        module_cfg["actor_critic"]["min_normalized_std"], dtype=torch.float32
    )

    actor = ActorCriticEncoder(module_cfg["actor_critic"]).cpu().eval()
    vae = VAEBlind(module_cfg["vae"]).cpu().eval()

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(loaded["model_state_dict"], strict=True)
    vae.load_state_dict(loaded["vae_state_dict"], strict=True)

    output_dir = (args.output_dir or default_output_dir(checkpoint)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "policy.onnx"
    export_and_verify(actor, vae, onnx_path)

    with policy_yaml_path.open(encoding="utf-8") as stream:
        policy_cfg = yaml.load(stream)
    policy_cfg["load_run"] = run_dir.name
    policy_cfg["checkpoint"] = checkpoint.name
    with (output_dir / "policy.yaml").open("w", encoding="utf-8") as stream:
        yaml.dump(policy_cfg, stream)

    print(f"[INFO] CPU-only export complete: {output_dir}")


if __name__ == "__main__":
    main()
