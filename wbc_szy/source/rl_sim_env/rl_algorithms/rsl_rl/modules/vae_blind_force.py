import torch
import torch.nn as nn
from rl_algorithms.rsl_rl.networks import MLP


class VAEBlindForce(nn.Module):
    """VAEBlind + 末端外力显式 head（force_control 专属）。

    与 ``VAEBlind`` 的唯一区别：多一个 ``encode_force`` head（``obs_force`` 维，默认 3），
    用于估计末端外力（projected COM yaw frame）。code 拼接顺序变为
    ``[vel, com, mass, force, latent]``，``num_estimator_out`` 相比 VAEBlind 多 3。

    不改 ``VAEBlind``，default/terrain 仍用原类，零影响。
    """

    def __init__(
        self,
        vae_cfg: dict,
    ):
        super().__init__()
        self.encoder = MLP(vae_cfg['encoder_in_dim'], vae_cfg['encoder_out_dim'], vae_cfg['encoder_hidden_dims'], vae_cfg['activation'], vae_cfg['activation'])

        self.encode_mean_latent = nn.Linear(vae_cfg['encoder_out_dim'], vae_cfg['encoder_head_dim_dict']['obs_latent'])
        self.encode_logvar_latent = nn.Linear(vae_cfg['encoder_out_dim'], vae_cfg['encoder_head_dim_dict']['obs_latent'])
        self.encode_vel = nn.Linear(vae_cfg['encoder_out_dim'], vae_cfg['encoder_head_dim_dict']['obs_vel'])
        self.encode_com = nn.Linear(vae_cfg['encoder_out_dim'], vae_cfg['encoder_head_dim_dict']['obs_com'])
        self.encode_mass = nn.Linear(vae_cfg['encoder_out_dim'], vae_cfg['encoder_head_dim_dict']['obs_mass'])
        self.encode_force = nn.Linear(vae_cfg['encoder_out_dim'], vae_cfg['encoder_head_dim_dict']['obs_force'])

        self.decoder = MLP(vae_cfg['decoder_in_dim'], vae_cfg['decoder_out_dim'], vae_cfg['decoder_hidden_dims'], vae_cfg['activation'], vae_cfg['activation'])

        print(f"VAE Encoder: {self.encoder}")
        print(f"VAE Encode Mean Latent: {self.encode_mean_latent}")
        print(f"VAE Encode Logvar Latent: {self.encode_logvar_latent}")
        print(f"VAE Encode Vel: {self.encode_vel}")
        print(f"VAE Encode Com: {self.encode_com}")
        print(f"VAE Encode Mass: {self.encode_mass}")
        print(f"VAE Encode Force: {self.encode_force}")
        print(f"VAE Decoder: {self.decoder}")

    def forward(self, obs_history, deterministic=False, decode=True):
        encoded = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(encoded)
        logvar_latent = self.encode_logvar_latent(encoded)
        code_vel = self.encode_vel(encoded)
        code_com = self.encode_com(encoded)
        code_mass = self.encode_mass(encoded)
        code_force = self.encode_force(encoded)

        logvar_latent = torch.clamp(logvar_latent, min=-10, max=10)
        code_latent = self.reparameterise(mean_latent, logvar_latent, deterministic)
        # side_info 顺序与 LocomotionPPOForce 的 adaboot gt_out 对齐：
        # [vel, com, mass, force] + latent
        side_info = torch.cat((code_vel, code_com, code_mass, code_force), dim=-1).detach()
        code = torch.cat((side_info, code_latent), dim=-1)

        # Rollout/action selection only consumes ``code`` and ``code_latent``.  The
        # decoder is kept on by default for PPO/VAE updates, but can be skipped in
        # rollout without changing policy inputs or RNG consumption (pure Linear+ELU).
        decoded = self.decoder(code) if decode else None

        return {
            "code": code,
            "code_vel": code_vel,
            "code_com": code_com,
            "code_mass": code_mass,
            "code_force": code_force,
            "code_latent": code_latent,
            "decoded": decoded,
            "mean_latent": mean_latent,
            "logvar_latent": logvar_latent
        }

    def reparameterise(self, mean, logvar, deterministic=False):
        if deterministic:
            return mean
        else:
            std = torch.exp(logvar * 0.5)
            code = mean + std * torch.randn_like(std)
            return code

    def act_inference(self, obs_history):
        out_dict = self.forward(
            obs_history, deterministic=True, decode=False
        )
        return out_dict["code"]
