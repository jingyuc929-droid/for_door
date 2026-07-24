import torch
import torch.nn as nn
from rl_algorithms.rsl_rl.networks import MLP


class VAEBlind(nn.Module):
    def __init__(self,
                 vae_cfg: dict,
                 ):
        super().__init__()
        self.encoder = MLP(vae_cfg['encoder_in_dim'], vae_cfg['encoder_out_dim'], vae_cfg['encoder_hidden_dims'], vae_cfg['activation'], vae_cfg['activation'])

        self.encode_mean_latent = nn.Linear(vae_cfg['encoder_out_dim'], vae_cfg['encoder_head_dim_dict']['obs_latent'])
        self.encode_logvar_latent = nn.Linear(vae_cfg['encoder_out_dim'], vae_cfg['encoder_head_dim_dict']['obs_latent'])
        self.encode_vel = nn.Linear(vae_cfg['encoder_out_dim'], vae_cfg['encoder_head_dim_dict']['obs_vel'])
        self.encode_com = nn.Linear(vae_cfg['encoder_out_dim'], vae_cfg['encoder_head_dim_dict']['obs_com'])
        self.encode_mass = nn.Linear(vae_cfg['encoder_out_dim'], vae_cfg['encoder_head_dim_dict']['obs_mass'])

        self.decoder = MLP(vae_cfg['decoder_in_dim'], vae_cfg['decoder_out_dim'], vae_cfg['decoder_hidden_dims'], vae_cfg['activation'])

        print(f"VAE Encoder: {self.encoder}")
        print(f"VAE Encode Mean Latent: {self.encode_mean_latent}")
        print(f"VAE Encode Logvar Latent: {self.encode_logvar_latent}")
        print(f"VAE Encode Vel: {self.encode_vel}")
        print(f"VAE Encode Com: {self.encode_com}")
        print(f"VAE Encode Mass: {self.encode_mass}")
        print(f"VAE Decoder: {self.decoder}")

    def forward(self, obs_history, deterministic=False, decode=True):
        encoded = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(encoded)
        logvar_latent = self.encode_logvar_latent(encoded)
        code_vel = self.encode_vel(encoded)
        code_com = self.encode_com(encoded)
        code_mass = self.encode_mass(encoded)

        logvar_latent = torch.clamp(logvar_latent, min=-10, max=10)
        code_latent = self.reparameterise(mean_latent, logvar_latent, deterministic)
        side_info = torch.cat((code_vel, code_com, code_mass), dim=-1).detach()
        code = torch.cat((side_info, code_latent), dim=-1)

        # The rollout policy only consumes ``code``/``code_latent``.  Keep the
        # decoder enabled by default for VAE updates, but allow action selection
        # and inference to skip this otherwise unused MLP.
        decoded = self.decoder(code) if decode else None

        return {
            "code": code,
            "code_vel": code_vel,
            "code_com": code_com,
            "code_mass": code_mass,
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
        out_dict = self.forward(obs_history, deterministic=True, decode=False)
        return out_dict["code"]
