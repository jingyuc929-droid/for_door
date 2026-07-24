import torch
import torch.nn as nn
from torch import Tensor
from rl_algorithms.rsl_rl.networks import MLP
from rl_algorithms.rsl_rl.modules.CNN import CNN
# from rl_algorithms.rsl_rl.modules.TransformerEncoder import TransformerEncoder
from rl_algorithms.rsl_rl.modules.GRU_encoder import GRUEncoder
from rsl_rl.utils import unpad_trajectories


class PIEEstimatorNet(nn.Module):
    def __init__(
        self,
        PIE_estimator_cfg: dict,
    ):
        super().__init__()
        self.PIE_estimator_cfg = PIE_estimator_cfg
        # mlp_encoder
        # (N,history_length*proprioceptive_obs_step_obs) -> (N,proprioceptive_obs_mlp_encoder_out_dim)
        self.proprioceptive_obs_mlp_encoder = MLP(
            PIE_estimator_cfg["proprioceptive_obs_mlp_encoder_in_dim"],
            PIE_estimator_cfg["proprioceptive_obs_mlp_encoder_out_dim"],
            PIE_estimator_cfg["proprioceptive_obs_mlp_encoder_hidden_dims"],
            PIE_estimator_cfg["activation"],
            PIE_estimator_cfg["activation"],
        )
        # cnn_encoder
        # (N,history_length(num_channels),depth_images_height,depth_images_width) -> (N, 16*2^(num_layers-1), H/2^(num_layers-1), W/2^(num_layers-1)), proprioceptive_obs_mlp_encoder_out_dim must be equal to 8*2^(num_layers-1)
        self.depth_images_cnn_encoder = CNN(
            PIE_estimator_cfg["depth_images_cnn_encoder_input_channels"],
            PIE_estimator_cfg["depth_images_cnn_encoder_output_dim"],
        )
        # # Transformer_encoder
        # self.transformer_encoder = TransformerEncoder(
        #     PIE_estimator_cfg["transformer_encoder_embed_dim"],
        #     PIE_estimator_cfg["transformer_encoder_num_heads"],
        #     PIE_estimator_cfg["transformer_encoder_feedforward_dim"],
        #     1,
        #     PIE_estimator_cfg["transformer_encoder_dropout"],
        #     PIE_estimator_cfg["transformer_encoder_activation"],
        # )
        # 暂时由MLP代替Transformer_encoder
        self.feature_fusion_mlp_encoder = MLP(
            PIE_estimator_cfg["feature_fusion_mlp_encoder_in_dim"],
            PIE_estimator_cfg["feature_fusion_mlp_encoder_out_dim"],
            PIE_estimator_cfg["feature_fusion_mlp_encoder_hidden_dims"],
            PIE_estimator_cfg["activation"],
            PIE_estimator_cfg["activation"],
        )
        # GRU_encoder
        self.gru_encoder = GRUEncoder(
            PIE_estimator_cfg["gru_encoder_in_dim"],
            PIE_estimator_cfg["gru_encoder_hidden_dim"],
            PIE_estimator_cfg["gru_encoder_out_dict"],
            PIE_estimator_cfg["gru_encoder_num_layers"],
        )
        # [num_layers, num_envs, hidden_dim]
        self.gru_out_hidden_states_last = None

        # MLP_decoder
        # proprioceptive_obs_decoder
        proprioceptive_obs_mlp_decoder_in_dim = sum(
            PIE_estimator_cfg["proprioceptive_obs_mlp_decoder_in_dim"].values()
        )
        self.proprioceptive_obs_mlp_decoder = MLP(
            proprioceptive_obs_mlp_decoder_in_dim,
            PIE_estimator_cfg["proprioceptive_obs_mlp_decoder_out_dim"],
            PIE_estimator_cfg["proprioceptive_obs_mlp_decoder_hidden_dims"],
            PIE_estimator_cfg["activation"],
            PIE_estimator_cfg["activation"],
        )
        # elevation_map_est_decoder
        elevation_map_est_mlp_decoder_in_dim = sum(
            PIE_estimator_cfg["elevation_map_est_mlp_decoder_in_dim"].values()
        )
        self.elevation_map_est_mlp_decoder = MLP(
            elevation_map_est_mlp_decoder_in_dim,
            PIE_estimator_cfg["elevation_map_est_mlp_decoder_out_dim"],
            PIE_estimator_cfg["elevation_map_est_mlp_decoder_hidden_dims"],
            PIE_estimator_cfg["activation"],
            PIE_estimator_cfg["activation"],
        )

    def reset_state(self):
        self.gru_out_hidden_states_last = None

    def reset_state_dones(self, dones: torch.Tensor):
        if self.gru_out_hidden_states_last is not None:
            self.gru_out_hidden_states_last[:, dones, :] = 0

    def get_gru_out_hidden_states_last(self):
        return self.gru_out_hidden_states_last

    def forward(
        self,
        proprioceptive_obs_history_time_series: torch.Tensor,
        depth_images_history_time_series: torch.Tensor,
        masks: torch.Tensor | None = None,
        gru_out_hidden_states: torch.Tensor | None = None,
        deterministic=False,
    ):
        '''
        Input:
        proprioceptive_obs_history_time_series: (Time_steps, batch, history_length*proprioceptive_obs_step_obs)
        depth_images_history_time_series: (Time_steps, batch, num_channels(history_length)*depth_images_height*depth_images_width)
        masks: (Time_steps, batch)
        gru_out_hidden_states: (num_layers, batch, hidden_dim)
        Output:
        PIE_estimator_net_out: dict[str, Tensor]
            proprioceptive_obs_decoded: (Time_steps, batch, proprioceptive_obs_mlp_decoder_out_dim)
            elevation_map_est_decoded: (Time_steps, batch, elevation_map_est_mlp_decoder_out_dim)
            gru_encoded_concat: (1*batch, sum(gru_encoder_hidden_dims.values()))
        '''
        # proprioceptive_obs_history: (Time_steps, batch, history_length*proprioceptive_obs_step_obs)
        # depth_images_history: (Time_steps, batch, num_channels(history_length), depth_images_height, depth_images_width)
        batch_size = proprioceptive_obs_history_time_series.shape[1]
        # reshape，使得输入满足mlp，CNN的输入要求，(Time_steps, batch, history_length*proprioceptive_obs_step_obs) -> (Time_steps*batch, history_length*proprioceptive_obs_step_obs)
        proprioceptive_obs_history_time_series = (
            proprioceptive_obs_history_time_series.reshape(
                -1, proprioceptive_obs_history_time_series.shape[-1]
            )
        )
        # reshape，使得输入满足CNN的输入要求，(Time_steps, batch, num_channels(history_length)*depth_images_height*depth_images_width) -> (Time_steps*batch, num_channels(history_length), depth_images_height, depth_images_width)
        depth_images_history_time_series = depth_images_history_time_series.reshape(
            -1, self.PIE_estimator_cfg["depth_images_cnn_encoder_input_channels"], self.PIE_estimator_cfg["depth_images_cnn_encoder_input_height"], self.PIE_estimator_cfg["depth_images_cnn_encoder_input_width"],
        )

        # (Time_steps*batch,history_length*proprioceptive_obs_step_obs) -> (Time_steps*batch,proprioceptive_obs_mlp_encoder_out_dim)
        proprioceptive_obs_encoded = self.proprioceptive_obs_mlp_encoder(
            proprioceptive_obs_history_time_series
        )

        # (Time_steps*batch,history_length(num_channels),depth_images_height,depth_images_width) -> (Time_steps*batch, depth_images_cnn_encoder_input_height*depth_images_cnn_encoder_input_width)
        depth_images_encoded = self.depth_images_cnn_encoder(
            depth_images_history_time_series
        )

        # (Time_steps*batch, proprioceptive_obs_mlp_encoder_out_dim + depth_images_cnn_encoder_input_height*depth_images_cnn_encoder_input_width) -> (Time_steps*batch, feature_fusion_mlp_encoder_in_dim)
        feature_fusion_input = torch.cat(
            (proprioceptive_obs_encoded, depth_images_encoded), dim=-1
        )

        # (Time_steps*batch, feature_fusion_mlp_encoder_in_dim) -> (Time_steps*batch, feature_fusion_mlp_encoder_out_dim)
        feature_fusion_encoded = self.feature_fusion_mlp_encoder(
            feature_fusion_input
        )

        # 暂时不使用transformer_encoder，使用transformer_input作为输入
        transformer_encoded: Tensor = feature_fusion_encoded

        # reshape，使得输入满足gru_encoder的输入要求，
        # (Time_steps*batch, feature_fusion_mlp_encoder_out_dim)
        # -> (Time_steps, batch, feature_fusion_mlp_encoder_out_dim)，其中 batch 维度由 -1 自动推断
        transformer_encoded = transformer_encoded.reshape(
            -1,
            batch_size,
            self.PIE_estimator_cfg["feature_fusion_mlp_encoder_out_dim"],
        )
        if masks is None:
            masks = torch.ones(transformer_encoded.shape[0], transformer_encoded.shape[1], device=transformer_encoded.device, dtype=torch.bool)

        if gru_out_hidden_states is not None:
            self.gru_out_hidden_states_last = gru_out_hidden_states.detach()
        elif self.gru_out_hidden_states_last is None or self.gru_out_hidden_states_last.shape[1] != transformer_encoded.shape[1]:
            self.gru_out_hidden_states_last = torch.zeros(self.PIE_estimator_cfg["gru_encoder_num_layers"], transformer_encoded.shape[1], self.gru_encoder.gru_encoder_hidden_dim, device=transformer_encoded.device)
        # (Time_steps, batch, feature_fusion_mlp_encoder_out_dim) -> (Time_steps, batch, gru_encoder_hidden_dict)
        gru_encoded, new_gru_out_hidden_states = self.gru_encoder(gru_input=transformer_encoded, deterministic=deterministic, gru_out_hidden_states=self.gru_out_hidden_states_last, masks=masks)
        # gru_encoded, new_gru_out_hidden_states = self.gru_encoder(gru_input=transformer_encoded, deterministic=deterministic, gru_out_hidden_states=self.gru_out_hidden_states_last)
        self.gru_out_hidden_states_last = new_gru_out_hidden_states.detach()

        # (Time_steps*batch, sum(gru_encoder_hidden_dims.values())) -> (Time_steps*batch, proprioceptive_obs_mlp_decoder_out_dim)
        proprioceptive_obs_decoded: Tensor = self.proprioceptive_obs_mlp_decoder(
            gru_encoded["hidden_all"].reshape(-1, self.PIE_estimator_cfg["PIE_estimator_net_gru_net_encoder_out"])
        )

        # (Time_steps*batch, gru_encoder_hidden_dims['elevation_map_est']) -> (Time_steps*batch, elevation_map_est_mlp_decoder_out_dim)
        elevation_map_est_decoded: Tensor = self.elevation_map_est_mlp_decoder(
            gru_encoded["elevation_map_est"].reshape(-1, self.PIE_estimator_cfg["elevation_map_est_mlp_decoder_in_dim"]["elevation_map_est"])
        )

        PIE_estimator_net_out = gru_encoded
        # (Time_steps*batch, *) -> (Time_steps, batch, *)
        PIE_estimator_net_out["proprioceptive_obs_decoded"] = proprioceptive_obs_decoded.reshape(-1, batch_size, self.PIE_estimator_cfg["proprioceptive_obs_mlp_decoder_out_dim"])
        PIE_estimator_net_out["elevation_map_est_decoded"] = elevation_map_est_decoded.reshape(-1, batch_size, self.PIE_estimator_cfg["elevation_map_est_mlp_decoder_out_dim"])
        
        # 将字典中的每个张量都按照masks进行unpad
        if masks is not None:
            for key, value in PIE_estimator_net_out.items():
                PIE_estimator_net_out[key] = unpad_trajectories(value, masks)
        
        # (Time_steps, batch, sum(gru_encoder_hidden_dims.values())) -> (1*batch, sum(gru_encoder_hidden_dims.values()))
        PIE_estimator_net_out["gru_encoded_concat"] = gru_encoded["hidden_all"][-1, :, :].reshape(-1, self.PIE_estimator_cfg["PIE_estimator_net_gru_net_encoder_out"])

        return PIE_estimator_net_out

    def act_inference(self, proprioceptive_obs_history, depth_images_history, gru_out_hidden_states: torch.Tensor | None = None, masks: torch.Tensor | None = None, deterministic=True):
        PIE_estimator_net_out = self.forward(
            proprioceptive_obs_history, depth_images_history, gru_out_hidden_states=gru_out_hidden_states, masks=masks, deterministic=deterministic
        )
        return PIE_estimator_net_out["gru_encoded_concat"], self.gru_out_hidden_states_last
