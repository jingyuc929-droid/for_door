from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.utils import unpad_trajectories
from rl_algorithms.rsl_rl.utils.log_print import (
    print_placeholder_end,
    print_placeholder_start,
)
from torch.amp import autocast


class VAEVit(nn.Module):
    def __init__(self,
                 env_num,
                 prop_obs_dim=45,
                 next_obs_dim=33,
                 prop_obs_his=2,
                 prop_token_num=2,
                 point_obs_his=2,
                 grid_num=1,
                 grid_point_num=1,
                 grid_lx=10,
                 grid_ly=10,
                 grid_d=0.1,
                 latent_out_dim=16,
                 vel_out_dim=3,
                 mass_out_dim=1,
                 heightmap_latent_out_dim=64,
                 footheight_latent_out_dim=24,
                 footheight_out_dim=100,
                 heightmap_out_dim=403,
                 d_model=64,
                 tf_nhead=4,
                 tf_num_layers=2,
                 gru_hidden=128,
                 gru_layers=2,
                 dt=0.02):
        super().__init__()
        self.dt = dt
        self.env_num = env_num
        self.prop_obs_dim = prop_obs_dim
        self.prop_obs_his = prop_obs_his
        self.prop_token_num = prop_token_num
        self.point_obs_his = point_obs_his
        self.grid_num = grid_num
        self.grid_point_num = grid_point_num
        self.grid_lx = grid_lx
        self.grid_ly = grid_ly
        self.grid_d = grid_d
        self.d_model = d_model
        self.next_obs_dim = next_obs_dim
        # proprioceptive
        # encoder
        self.proprioceptive_encoder = nn.Sequential(
            nn.Linear(prop_obs_dim * prop_obs_his // prop_token_num, 128),
            # nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(128, d_model),
            # nn.LayerNorm(d_model),
            nn.ELU(),
        )
        # exteroceptive
        # pointnet
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )
        # center pos encoder
        self.center_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # # time pe
        # self.time_embedding = nn.Embedding(prop_token_num, d_model)

        # cls
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # self.cls_pe = nn.Parameter(torch.zeros(1, 1, d_model))

        # 定义两个类型：0=prop, 1=point
        self.type_embedding = nn.Embedding(2, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=tf_nhead,
            dim_feedforward=d_model * 4,
            # dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=tf_num_layers
        )
        # self.tf_norm = nn.LayerNorm(d_model)
        # self.tf_dropout = nn.Dropout(0.1)

        # GRU
        num_tokens = self.prop_token_num + self.point_obs_his * self.grid_num
        # num_tokens = 3
        self.gru = nn.GRU(
            input_size=num_tokens * d_model,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=False,
            # dropout=0.1,
        )
        # self.gru_norm = nn.LayerNorm(gru_hidden)
        # self.gru_dropout = nn.Dropout(0.1)

        self._gru_last_h = None
        # gru head
        self.gru_head_obs_mean_latent = nn.Linear(gru_hidden, latent_out_dim)
        self.gru_head_obs_logvar_latent = nn.Linear(gru_hidden, latent_out_dim)
        self.gru_head_mean_heightmap_latent = nn.Linear(gru_hidden, heightmap_latent_out_dim)
        self.gru_head_logvar_heightmap_latent = nn.Linear(gru_hidden, heightmap_latent_out_dim)

        self.gru_head_vel_mean = nn.Linear(gru_hidden, vel_out_dim)
        self.gru_head_vel_logvar = nn.Linear(gru_hidden, vel_out_dim)
        self.gru_head_mass = nn.Linear(gru_hidden, mass_out_dim)
        self.gru_head_mass_logvar = nn.Linear(gru_hidden, mass_out_dim)
        self.gru_head_footheight_mean = nn.Linear(gru_hidden, footheight_out_dim)
        self.gru_head_footheight_logvar = nn.Linear(gru_hidden, footheight_out_dim)

        # decoder
        self.prop_obs_decoder = nn.Sequential(
            nn.Linear(latent_out_dim, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, self.next_obs_dim),
        )
        self.heightmap_decoder = nn.Sequential(
            nn.Linear(heightmap_latent_out_dim, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, heightmap_out_dim),
        )
        # self.footheight_rough_decoder = nn.Sequential(
        #     nn.Linear(footheight_latent_out_dim, 64),
        #     nn.ELU(),
        #     nn.Linear(64, 128),
        #     nn.ELU(),
        #     nn.Linear(128, footheight_out_dim),
        # )
        # self.footheight_fine_decoder = nn.Sequential(
        #     nn.Linear(footheight_out_dim, 128),
        #     nn.ELU(),
        #     nn.Linear(128, 128),
        #     nn.ELU(),
        #     nn.Linear(128, footheight_out_dim),
        # )

        print_placeholder_start("VAEVit")
        print(f"VAEVit proprioceptive_encoder: {self.proprioceptive_encoder}")
        print(f"VAEVit transformer: {self.transformer}")
        print(f"VAEVit gru: {self.gru}")
        print(f"VAEVit obs_decoder: {self.prop_obs_decoder}")
        print(f"VAEVit heightmap_decoder: {self.heightmap_decoder}")
        print(f"VAEVit gru_head_obs_mean_latent: {self.gru_head_obs_mean_latent}")
        print(f"VAEVit gru_head_obs_logvar_latent: {self.gru_head_obs_logvar_latent}")
        print(f"VAEVit gru_head_mean_heightmap_latent: {self.gru_head_mean_heightmap_latent}")
        print(f"VAEVit gru_head_logvar_heightmap_latent: {self.gru_head_logvar_heightmap_latent}")
        print(f"VAEVit gru_head_vel_mean: {self.gru_head_vel_mean}")
        print(f"VAEVit gru_head_vel_logvar: {self.gru_head_vel_logvar}")
        print(f"VAEVit gru_head_mass: {self.gru_head_mass}")
        print(f"VAEVit gru_head_mass_logvar: {self.gru_head_mass_logvar}")
        print(f"VAEVit gru_head_footheight_mean: {self.gru_head_footheight_mean}")
        print(f"VAEVit gru_head_footheight_logvar: {self.gru_head_footheight_logvar}")
        print_placeholder_end()

    def sinusoidal_pe(self, time: torch.Tensor, d_model: int) -> torch.Tensor:
        """
        time: (N,) 一维张量，每个元素是真实的时间戳或累积时间
        returns: (N, d_model) 的 sin–cos PE
        """
        N = time.size(0)
        pe = torch.zeros(N, d_model, device=time.device)
        # 计算每个偶数维度的分母
        div_term = torch.exp(torch.arange(0, d_model, 2, device=time.device) *
                             -(9.210340372 / d_model))  # (d_model/2,)
        pe[:, 0::2] = torch.sin(time.unsqueeze(1) * div_term).to(time.device)
        pe[:, 1::2] = torch.cos(time.unsqueeze(1) * div_term).to(time.device)
        return pe  # (N, d_model)

    def reset_state(self):
        """
        reset the hidden state of the GRU
        """
        self._gru_last_h = None

    def reset_state_dones(self, dones: torch.Tensor):
        """
        reset the hidden state of the GRU
        """
        if self._gru_last_h is not None:
            self._gru_last_h[:, dones, :] = 0

    def get_gru_last_h(self):
        return self._gru_last_h

    def forward(self,
                obs_history: torch.Tensor,
                point_history: torch.Tensor,
                deterministic: bool = False):
        return self.cenet_forward(obs_history, point_history, deterministic)

    def cenet_forward(self,
                      prop_history: torch.Tensor,  # (T,B,prop_his*prop_dim)
                      point_history: torch.Tensor,  # (T,B,point_his*grid_num*grid_point_num,3)
                      hidden_states: torch.Tensor | None = None,  # (L,B,gru_hidden)
                      masks: torch.Tensor | None = None,  # (T,B)
                      print_info: bool = False,
                      deterministic: bool = False):
        # print(point_history)
        # print(f"vae vit debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if print_info:
            print(f"vae vit debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("prop_history", prop_history)
            print("point_history", point_history)
            print("hidden_states", hidden_states)
            print("masks", masks)

        T = prop_history.size(0)  # time
        B = prop_history.size(1)  # batch
        P = self.prop_token_num  # prop token_num
        Q = self.point_obs_his  # point_his
        M = self.grid_num       # grid_num
        K = self.grid_point_num        # grid_point_num

        # print("prop_history", prop_history.shape)
        prop_history = prop_history.reshape(T, B, P, -1)
        # print(f"prop_history.shape: {prop_history.shape}")
        point_history = point_history.reshape(T, B * Q, M * K, 3)
        # print(f"point_history.shape: {point_history.shape}")

        if hidden_states is not None:
            self._gru_last_h = hidden_states.clone().detach()
        elif self._gru_last_h is None or self._gru_last_h.size(1) != B:
            # print("B", B)
            self._gru_last_h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size,
                                           device=next(self.parameters()).device)

        if masks is None:
            masks = torch.ones(T, B, dtype=torch.bool, device=prop_history.device)

        # === 1) Proprioceptive  ===
        # 1.1 MLP encode each frame -> (N, P, d_model)
        z_prop = self.proprioceptive_encoder(prop_history)  # -> (T, B, P, d_model)
        type_idx_prop = torch.zeros(P, dtype=torch.long, device=z_prop.device)  # (1,)
        type_emb_prop = self.type_embedding(type_idx_prop)                     # (1, d_model)
        z_prop = z_prop + type_emb_prop.view(1, 1, P, self.d_model)
        # print(f"z_prop.shape: {z_prop.shape}")
        # 1.2 Time PE
        # idx_prop = torch.arange(P, device=prop_history.device)       # [0,1,…,P-1]
        # pe_p = self.time_embedding(idx_prop)                   # (P, d_model)
        # print(f"pe_p.shape: {pe_p.shape}")
        # z_prop = z_prop + pe_p.unsqueeze(0)                       # (T, B, P, d_model)
        # print(f"z_prop.shape+pe: {z_prop.shape}")
        # times_prop = torch.arange(P, device=prop_history.device, dtype=torch.float32) * self.dt  # (P,)
        # pe_real = self.sinusoidal_pe(times_prop, self.d_model)       # (P, d_model)
        # z_prop = z_prop + pe_real.unsqueeze(0)                       # (T, B, P, d_model)

        # === 2) Exteroceptive  ===
        # 2.1 flatten environment/frame/center to batch dimension, prepare Conv1d input
        #    (T, B, Q, M, K, 3) -> (T*B*Q*M, K, 3) -> (T*B*Q*M, 3, K)
        # print(f"point_history.shape: {point_history.shape}")

        # centers(T, B*Q, M, 3), point_out(T, B*Q, M, K, 3), mask_grid(T, B*Q, M), mask_point(T, B*Q, M, K)
        centers, point_out, mask_grid, mask_point = self.grid_partition_xy(point_history, self.grid_lx, self.grid_ly, self.grid_d, K)

        pts = point_out.reshape(T * B * Q * M, K, 3).permute(0, 2, 1)  # (T*B*Q*M, 3, K)
        centers_local = centers.reshape(T * B * Q * M, 3)             # broadcast-safe
        pts_rel = pts - centers_local.unsqueeze(2)              # (T*B*Q*M, 3, K)
        mask_flat = mask_point.reshape(-1, K).unsqueeze(1)      # (T*B*Q*M, 1, K)
        mask_grid_flat = mask_grid.reshape(-1).unsqueeze(1).unsqueeze(2)             # (N_all, 1, 1)
        mask_tb = masks.view(T, B, 1, 1).expand(-1, -1, Q, M)  # (T, B, Q, M)
        mask_tb_flat = mask_tb.reshape(-1).unsqueeze(1).unsqueeze(2)             # (N_all, 1, 1)
        mask_grid_b = mask_grid_flat.expand(-1, 1, K)    # (N_all,1,K)
        mask_tb_b = mask_tb_flat  .expand(-1, 1, K)    # (N_all,1,K)
        combined_mask = mask_flat & mask_grid_b & mask_tb_b  # (N_all,1,K), bool

        pts_rel = pts_rel * combined_mask.float()            # (N_all,3,K)

        # 2.2 Shared MLP + max pool -> (T*B*Q*M, d_model)
        if print_info:
            print("pts_rel", pts_rel)
        feat_full = self.shared_mlp(pts_rel)                    # -> (T*B*Q*M, d_model, K)
        if print_info:
            print("feat_full", feat_full)
        # # print(f"feat_full.shape: {feat_full.shape}")
        # std = feat_full.std(dim=2, keepdim=True, unbiased=False)     # (T*B*Q*M, d_model, 1)
        # conf = 1.0 - torch.tanh(std)                                 # (T*B*Q*M, d_model, 1)
        # feat_full = feat_full * conf                                 # (T*B*Q*M, d_model, K)
        mask_used = combined_mask.squeeze(1)  # (N_all, K)
        mask_us = mask_used.unsqueeze(1)      # (N_all, 1, K)
        # print("feat_full: ", feat_full)
        feat_full = feat_full.masked_fill(
            ~mask_us,
            float(0)
        )

        feat, _ = feat_full.max(dim=2)                    # -> (T*B*Q*M, d_model)
        # print(f"feat.shape: {feat.shape}")
        # 2.3 reshape back to (T, B, Q, M, d_model)
        z_pt = feat.reshape(T, B, Q, M, self.d_model)        # (T, B, Q, M, d_model)
        # 2.4 add spatial PE (the first one is the center point)
        centers_pool = centers.reshape(T, B, Q, M, 3)
        pe_sp = self.center_mlp(centers_pool)                     # (T, B, Q, M, d_model)
        z_pt = z_pt + pe_sp                                   # (T, B, Q, M, d_model)
        # print(f"z_pt.shape: {z_pt.shape}")
        type_idx_pt = torch.ones(Q * M, dtype=torch.long, device=z_pt.device)     # (Q*M,)
        type_emb_pt = self.type_embedding(type_idx_pt)                         # (Q*M, d_model)
        z_pt = z_pt + type_emb_pt.view(1, 1, Q * M, self.d_model)
        # print("z_pt: ", z_pt)
        # assert False
        # 2.5 add time PE
        # stride = max(1, P // Q)         # step
        # idx_pc = torch.arange(0, P, step=stride, device=prop_history.device)[:Q]
        # # ensure idx_pc length is exactly Q
        # assert idx_pc.numel() == Q, "idx_pc length should be equal to Q"
        # # print("idx_pc: ", idx_pc)
        # pe_q = self.time_embedding(idx_pc).reshape(1, Q, 1, self.d_model)
        # # print(f"pe_q.shape: {pe_q.shape}")
        # # print(pe_q)
        # z_pt = z_pt + pe_q              # (T, B, Q, M, d_model)

        # times_pt = (torch.arange(Q, device=prop_history.device, dtype=torch.float32)
        #             * (stride * self.dt))
        # times_pt = times_pt.repeat_interleave(M)           # (Q*M,)
        # pe_real_q = self.sinusoidal_pe(times_pt, self.d_model)  # (Q*M, d_model)
        # pe_real_q = pe_real_q.view(1, 1, Q, M, self.d_model)            # (1,1,Q,M,D)
        # z_pt = z_pt + pe_real_q                      # (T, B, Q, M, d_model)

        # 2.6 flatten to sequence -> (T, B, Q*M, d_model)
        z_pt = z_pt.reshape(T, B, Q * M, self.d_model)

        # === 3) concatenate all tokens and add CLS ===
        # 3.1 concatenate proprioceptive and exteroceptive tokens
        tokens = torch.cat([z_prop, z_pt], dim=2)            # -> (T, B, P+Q*M, d_model)

        # 3.2 insert CLS token at the beginning
        # cls = (self.cls_token + self.cls_pe).expand(T, B, -1, -1)  # (T, B, 1, d_model)
        # tokens = torch.cat([cls, tokens], dim=2)                 # -> (T, B, 1+P+Q*M, d_model)

        # === 4) Transformer encode ===
        # kpm = (~masks).view(T * B)[:, None].expand(-1, tokens.shape[2])
        # all_pad = kpm.all(dim=1)            # bool  (T*B,)  True 表示这一整行全 pad
        # if all_pad.any():
        #     kpm[all_pad, 0] = False         # 把第 0 个 token（即 CLS）强制保留
        # _, _, L, D = tokens.shape
        T, B, L, D = tokens.shape
        kpm = (~masks)             # (T, B)  True 表示 pad
        kpm = kpm.reshape(T * B, 1).expand(-1, L)  # (T*B, S)
        if print_info:
            print("kpm", kpm)
            print("tokens", tokens)
        tf_out = self.transformer(tokens.reshape(T * B, L, D), src_key_padding_mask=kpm)        # (T*B, 1+P+Q*M, d_model)
        if print_info:
            print("tf_out", tf_out)
        tf_out = tf_out.reshape(T, B, L, D)

        # x = tokens.reshape(T * B, L, D)  # shape (T*B, L, D)
        # for layer in self.transformer.layers:
        #     def custom_forward(inp: torch.Tensor):
        #         return layer(inp, src_key_padding_mask=kpm)
        #     x = checkpoint(custom_forward, x, use_reentrant=False)
        # tf_out = x.reshape(T, B, L, D)

        # tf_out = self.tf_norm(tf_out)
        # tf_out = self.tf_dropout(tf_out)
        # cls+pool
        # cls = tf_out[:, :, 0]                        # (T, B, d)
        # prop_out = tf_out[:, :, 1:P + 1].reshape(T, B, P * D)                # (T, B, P, d)
        # point_out = tf_out[:, :, P + 1:].max(dim=2).values     # (T, B, d)
        # step_embed = torch.cat([cls, prop_out, point_out], dim=-1)   # (T, B, (P+2)*d)

        # 把所有 token flatten，保留顺序：(T, B, L, d_model) -> (T, B, L*d_model)
        # all tokens
        # T, B, L, D = tf_out.shape
        step_embed = tf_out.reshape(T, B, L * D)      # (T, B, num_tokens * d_model)

        # === 5) GRU encode ===
        lens = masks.sum(dim=0).clamp(min=1)          # (B,) ≥1
        lens_cpu = lens.to(torch.long).cpu()                # keep on CPU
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            step_embed, lens_cpu, enforce_sorted=False)
        if print_info:
            print("packed", packed)
        gru_out, new_h = self.gru(packed, self._gru_last_h)
        if print_info:
            print("gru_out", gru_out)
            print("new_h", new_h)
        gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, total_length=T)
        # save for next forward
        self._gru_last_h = new_h.detach()  # detach to prevent cross-sequence backpropagation
        # gru_out = self.gru_norm(gru_out)
        # gru_out = self.gru_dropout(gru_out)
        gru_out = gru_out.reshape(T, B, -1)

        gru_out = unpad_trajectories(gru_out, masks)

        # print(f"gru_out.shape: {gru_out.shape}")
        # === 6) multi-head VAE branches ===
        mean_obs = self.gru_head_obs_mean_latent(gru_out)       # (T, B, L, latent_out_dim)
        logvar_obs = self.gru_head_obs_logvar_latent(gru_out)
        mean_hmap = self.gru_head_mean_heightmap_latent(gru_out)
        logvar_hmap = self.gru_head_logvar_heightmap_latent(gru_out)

        mean_v = self.gru_head_vel_mean(gru_out)
        logvar_v = self.gru_head_vel_logvar(gru_out)
        mean_m = self.gru_head_mass(gru_out)
        logvar_m = self.gru_head_mass_logvar(gru_out)
        mean_fh = self.gru_head_footheight_mean(gru_out)
        logvar_fh = self.gru_head_footheight_logvar(gru_out)

        # clamp logvar
        # with autocast(device_type='cuda', enabled=False):
        logvar_obs = torch.clamp(logvar_obs, min=-10, max=10)
        logvar_hmap = torch.clamp(logvar_hmap, min=-10, max=10)
        logvar_v = torch.clamp(logvar_v, min=-10, max=10)
        logvar_m = torch.clamp(logvar_m, min=-10, max=10)
        logvar_fh = torch.clamp(logvar_fh, min=-10, max=10)

        # reparameterise per token
        code_obs_latent = self.reparameterise(mean_obs, logvar_obs, deterministic)  # (T, B, L, latent)
        code_hmap_latent = self.reparameterise(mean_hmap, logvar_hmap, deterministic)
        code_v = self.reparameterise(mean_v, logvar_v, deterministic)
        code_m = self.reparameterise(mean_m, logvar_m, deterministic)
        code_fh = self.reparameterise(mean_fh, logvar_fh, deterministic)

        # concat all latent channels
        code = torch.cat([code_v, code_m, code_fh, code_obs_latent, code_hmap_latent], dim=-1)  # (T, B, L, sum_latent)
        # obs_decoded_input = torch.cat([code_v, code_m, code_obs_latent], dim=-1)
        # === 7) decode ===
        prop_obs_decoded = self.prop_obs_decoder(code_obs_latent)         # (T, B, L, prop_dim)
        heightmap_decoded = self.heightmap_decoder(code_hmap_latent)   # (T, B, L, heightmap_out_dim)
        # footheight_decoded = self.footheight_decoder(code_fh)   # (T, B, L, footheight_out_dim)
        # footheight_rough_decoded = self.footheight_rough_decoder(code_fh)   # (T, B, L, footheight_out_dim)
        # footheight_fine_decoded = self.footheight_fine_decoder(footheight_rough_decoded)   # (T, B, L, footheight_out_dim)

        return {
            "code": code,
            "code_vel": code_v,
            "code_mass": code_m,
            "code_obs_latent": code_obs_latent,
            "code_heightmap_latent": code_hmap_latent,
            "code_footheight": code_fh,
            "prop_obs_decoded": prop_obs_decoded,
            "heightmap_decoded": heightmap_decoded,
            # "footheight_rough_decoded": footheight_rough_decoded,
            # "footheight_fine_decoded": footheight_fine_decoded,
            "mean_obs": mean_obs,
            "logvar_obs": logvar_obs,
            "mean_hmap": mean_hmap,
            "logvar_hmap": logvar_hmap,
        }

    def reparameterise(self,
                       mean: torch.Tensor,
                       logvar: torch.Tensor,
                       deterministic: bool = False):

        if deterministic:
            return mean
        else:
            var = torch.exp(logvar * 0.5)
            code_temp = torch.randn_like(var)
            code = mean + var * code_temp
            return code

    def act_inference(self,
                      obs_history: torch.Tensor,
                      point_history: torch.Tensor,
                      ):
        dict_out = self.cenet_forward(obs_history, point_history, deterministic=True)
        return dict_out["code"], dict_out["heightmap_decoded"], dict_out["code_footheight"]

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

    def grid_partition_xy(
        self,
        points: torch.Tensor,  # (T, B, N, 3)
        lx: float, ly: float,  # 半长半宽
        d: float,              # 网格边长
        p: int                 # 每格采样点数
    ):
        nx = int((2 * lx) // d)
        ny = int((2 * ly) // d)
        M = nx * ny

        # 1. 计算格子 ID g ∈ [0,M)
        x_shift = points[..., 0] + lx   # (T, B, N)
        y_shift = points[..., 1] + ly
        ix = torch.clamp((x_shift / d).floor().long(), 0, nx - 1)
        iy = torch.clamp((y_shift / d).floor().long(), 0, ny - 1)
        g = ix + iy * nx               # (T, B, N)

        # 2. 构造属于每个格子的 one‑hot eq 掩码
        # eq[t,b,n,m] = True 表示第 n 个点属于第 m 个格子
        grid_ids = torch.arange(M, device=points.device)
        eq = g.unsqueeze(-1) == grid_ids.view(1, 1, 1, -1)  # (T, B, N, M)

        # —— 在这里计算每个格子的真实点平均中心 —— #
        # 把 points 扩一个格子维度后再乘 eq，然后对 N 求和
        pts_expand = points.unsqueeze(3)                   # (T, B, N, 1, 3)
        weighted = pts_expand * eq.unsqueeze(-1).float()   # (T, B, N, M, 3)
        sum_xyz = weighted.sum(dim=2)                      # (T, B, M, 3)
        cnt = eq.sum(dim=2).clamp(min=1).unsqueeze(-1)  # (T, B, M, 1)，至少 1 避免除 0
        # with autocast(device_type='cuda', enabled=False):
        centers = sum_xyz / cnt                             # (T, B, M, 3)
        # 对于原本就没有任何点的格子，cnt==0 时我们把分母 clamp 为 1，sum_xyz=0 => centers=(0,0,0)

        # 3. 随机键 & masked_rnd（和之前一致）
        rnd = torch.rand_like(x_shift)  # (T, B, N)
        masked_rnd = torch.where(
            eq, rnd.unsqueeze(-1),
            torch.ones_like(rnd.unsqueeze(-1))
        )  # (T, B, N, M)

        # 4. 对每格做 top‑p
        values, idx = masked_rnd.topk(p, dim=2, largest=False)  # (T, B, p, M)

        # 5. Gather 出坐标
        pts_exp2 = points.unsqueeze(3).expand(-1, -1, -1, M, -1)    # (T, B, N, M, 3)
        idx_us = idx.unsqueeze(-1).expand(-1, -1, -1, -1, 3)      # (T, B, p, M, 3)
        pts_mod = torch.gather(pts_exp2, 2, idx_us)            # (T, B, p, M, 3)
        pts_out = pts_mod.permute(0, 1, 3, 2, 4)               # (T, B, M, p, 3)

        # 6. 构造 Mask 并填充
        mask_point = (values < 1.0).permute(0, 1, 3, 2)             # (T, B, M, p)
        mask_grid = mask_point.any(dim=-1)                    # (T, B, M)
        pts_out = pts_out * mask_point.unsqueeze(-1).float()

        # 最终多返回一个 centers
        return centers, pts_out, mask_grid, mask_point
