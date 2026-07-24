from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_algorithms.rsl_rl.utils.log_print import (
    print_placeholder_end,
    print_placeholder_start,
)


class VAE(nn.Module):
    def __init__(self, cenet_in_dim, cenet_out_dim):
        super().__init__()

        self.activation = nn.ELU()

        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim, 128),
            self.activation,
            nn.Linear(128, 64),
            self.activation,
        )
        self.encode_mean_latent = nn.Linear(64, cenet_out_dim - 4)
        self.encode_logvar_latent = nn.Linear(64, cenet_out_dim - 4)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)
        self.encode_mean_mass = nn.Linear(64, 1)
        self.encode_logvar_mass = nn.Linear(64, 1)

        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim, 64),
            self.activation,
            nn.Linear(64, 128),
            self.activation,
            nn.Linear(128, 45),
        )

        print_placeholder_start("VAE")
        print(f"VAE: {self.encoder}")
        print(f"VAE: {self.encode_mean_latent}")
        print(f"VAE: {self.encode_logvar_latent}")
        print(f"VAE: {self.encode_mean_vel}")
        print(f"VAE: {self.encode_logvar_vel}")
        print(f"VAE: {self.encode_mean_mass}")
        print(f"VAE: {self.encode_logvar_mass}")
        print_placeholder_end()

    def forward(self):
        raise NotImplementedError

    def soft_clip(self, x, low, high):
        # 第一步：平滑下界
        x = low + F.softplus(x - low)
        # 第二步：平滑上界
        x = high - F.softplus(high - x)
        return x

    def cenet_forward(self, obs_history, deterministic=False):
        # encode
        encoded = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(encoded)
        logvar_latent = self.encode_logvar_latent(encoded)
        mean_vel = self.encode_mean_vel(encoded)
        logvar_vel = self.encode_logvar_vel(encoded)
        mean_mass = self.encode_mean_mass(encoded)
        logvar_mass = self.encode_logvar_mass(encoded)

        logvar_latent = torch.clamp(logvar_latent, min=-10, max=10)
        logvar_vel = torch.clamp(logvar_vel, min=-10, max=10)
        logvar_mass = torch.clamp(logvar_mass, min=-10, max=10)
        # logvar_latent = self.soft_clip(logvar_latent, low=-10.0, high=10.0)
        # logvar_vel = self.soft_clip(logvar_vel, low=-10.0, high=10.0)

        # reparameterise
        code_latent = self.reparameterise(mean_latent, logvar_latent, deterministic)
        code_vel = self.reparameterise(mean_vel, logvar_vel, deterministic)
        code_mass = self.reparameterise(mean_mass, logvar_mass, deterministic)

        code = torch.cat((code_vel, code_mass, code_latent), dim=-1)

        # decode
        decoded = self.decoder(code)

        return (
            code,
            code_vel,
            code_mass,
            code_latent,
            decoded,
            mean_vel,
            logvar_vel,
            mean_latent,
            logvar_latent,
            mean_mass,
            logvar_mass,
        )

    def reparameterise(self, mean, logvar, deterministic=False):
        if deterministic:
            return mean
        else:
            var = torch.exp(logvar * 0.5)
            code_temp = torch.randn_like(var)
            code = mean + var * code_temp
            return code

    def act_inference(self, obs_history):
        (
            code,
            code_vel,
            code_mass,
            code_latent,
            decoded,
            mean_vel,
            logvar_vel,
            mean_latent,
            logvar_latent,
            mean_mass,
            logvar_mass,
        ) = self.cenet_forward(obs_history, deterministic=True)
        return code

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
