import torch
import torch.nn as nn
from rl_algorithms.rsl_rl.networks import MLP


class EstimatorNet(nn.Module):
    def __init__(self,
                 estimator_cfg: dict,
                 ):
        super().__init__()
        self.est_encoder = MLP(estimator_cfg['est_encoder_in_dim'], estimator_cfg['est_encoder_out_dim'], estimator_cfg['est_encoder_hidden_dims'], estimator_cfg['activation'], estimator_cfg['activation'])
        self.est_encode_lin_vel = nn.Linear(estimator_cfg['est_encoder_out_dim'], estimator_cfg['est_encoder_head_dim_dict']['obs_base_lin_vel'])
        self.est_encode_contact_flags = nn.Linear(estimator_cfg['est_encoder_out_dim'], estimator_cfg['est_encoder_head_dim_dict']['obs_foot_contact_flags'])
        print(f"Estimator Net: {self.est_encoder}")
        print(f"Estimator Net Encode Lin Vel: {self.est_encode_lin_vel}")
        print(f"Estimator Net Encode Contact Flags: {self.est_encode_contact_flags}")

    def forward(self, obs_history, deterministic=False):
        est_encoded = self.est_encoder(obs_history)
        est_base_lin_vel = self.est_encode_lin_vel(est_encoded)
        est_foot_contact_flags = self.est_encode_contact_flags(est_encoded)
        est_state_info = torch.cat((est_base_lin_vel, est_foot_contact_flags), dim=-1)

        return {
            "est_encoded": est_encoded,
            "est_base_lin_vel": est_base_lin_vel,
            "est_foot_contact_flags": est_foot_contact_flags,
            "est_state_info": est_state_info,
        }

    def reparameterise(self, mean, logvar, deterministic=False):
        if deterministic:
            return mean
        else:
            std = torch.exp(logvar * 0.5)
            code = mean + std * torch.randn_like(std)
            return code

    def act_inference(self, obs_history):
        out_dict = self.forward(obs_history, deterministic=True)
        return out_dict["est_state_info"]
