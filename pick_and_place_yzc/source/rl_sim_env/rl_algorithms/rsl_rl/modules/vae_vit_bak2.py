from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.utils import unpad_trajectories


class UNetRefine(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: list[int] = [16, 32, 64]):
        """
        U‑Net‑based Refine Decoder（支持恢复奇数尺寸，无需 center_crop）
        - 下采样：MaxPool2d(k=2, s=2, ceil_mode=True)
        - 上采样：ConvTranspose2d 逐层指定 padding & output_padding
        - 跳跃连接：直接拼接，不裁剪
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # Encoder
        self.down_blocks = nn.ModuleList()
        prev_ch = in_channels
        for ch in features:
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(prev_ch, ch, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True),
            ))
            prev_ch = ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_ch, prev_ch, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(prev_ch, prev_ch, 3, 1, 1), nn.ReLU(inplace=True),
        )

        # Decoder：分别为 三个上采样 层 定义 (padding, output_padding)
        # 从最深层到最浅层对应的空间跳跃连接尺寸分别是：
        #   skip3: 16×7  ← 输入 8×4 上采样 →16×7
        #   skip2: 31×13 ← 输入16×7 上采样→31×13
        #   skip1: 61×25 ← 输入31×13 上采样→61×25
        self.up_pads = [
            (0, 1),  # 对应 第一层上采样 (8→16, 4→7)
            (1, 1),  # 对应 第二层上采样 (16→31,7→13)
            (1, 1),  # 对应 第三层上采样 (31→61,13→25)
        ]
        self.up_opads = [
            (0, 1),
            (1, 1),
            (1, 1),
        ]

        self.up_transposes = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        prev_ch = features[-1]
        for idx, ch in enumerate(reversed(features)):
            pad_h, pad_w = self.up_pads[idx]
            op_h, op_w = self.up_opads[idx]

            # 上采样：kernel=2,stride=2，对应级别的 pad/opad
            self.up_transposes.append(
                nn.ConvTranspose2d(prev_ch, ch,
                                   kernel_size=2, stride=2,
                                   padding=(pad_h, pad_w),
                                   output_padding=(op_h, op_w))
            )
            # 拼接后通道 = ch*2，再两层 3×3 卷积 + ReLU
            self.up_blocks.append(nn.Sequential(
                nn.Conv2d(ch * 2, ch, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True),
            ))
            prev_ch = ch

        # 最后 1×1 卷积输出
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        # Encoder 路径
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder 路径（无需 center_crop）
        for trans, up_block, skip in zip(self.up_transposes,
                                         self.up_blocks,
                                         reversed(skips)):
            x = trans(x)
            # 此时 x 的 H×W 恰好 == skip 的 H×W
            x = torch.cat([skip, x], dim=1)
            x = up_block(x)

        return self.final_conv(x)


class HeightmapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=6, padding=3),  # (61, 25) -> (11, 5)
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (11, 5) -> (11, 5)
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (11, 5) -> (11, 5)
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
          x: (K, 1, H_in, W_in)  单通道高程图
        输出:
          z_map: (K, N_tok, d_model)  空间 token 序列
        """
        x = self.encoder(x)
        K, C, H, W = x.shape
        x = x.view(K, C, H * W)
        x = x.permute(0, 2, 1)
        return x


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        # 对 C 维标准化
        self.norm1 = nn.LayerNorm(hidden_dim)
        # 跨 token 的 MLP（输入(B, C, T)→输出(B, C, T)）
        self.token_mixing = nn.Sequential(
            nn.Linear(num_tokens, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, num_tokens),
        )
        # 第二次标准化
        self.norm2 = nn.LayerNorm(hidden_dim)
        # 跨通道的 MLP（输入(B, T, C)→输出(B, T, C)）
        self.channel_mixing = nn.Sequential(
            nn.Linear(hidden_dim, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, hidden_dim),
        )

    def forward(self, x):
        # x: (B, T, C)
        # 1) Token-mixing part
        y = self.norm1(x)                # →(B, T, C)，标准化最后 C 维
        y = y.permute(0, 2, 1)           # →(B, C, T)，准备跨 token MLP
        y = self.token_mixing(y)         # →(B, C, T)
        y = y.permute(0, 2, 1)           # →(B, T, C)
        x = x + y                        # 残差连接

        # 2) Channel-mixing part
        z = self.norm2(x)                # →(B, T, C)
        z = self.channel_mixing(z)       # →(B, T, C)
        return x + z                     # 残差连接


class VAEVit(nn.Module):
    def __init__(self,
                 env_num,
                 prop_obs_in_dim=225,
                 prop_decoder_out_dim=33,
                 heightmap_decoder_out_h_dim=61,
                 heightmap_decoder_out_w_dim=25,
                 footheight_decoder_out_dim=36,
                 heightmap_latent_out_dim=128,
                 footheight_latent_out_dim=16,
                 obs_latent_out_dim=16,
                 vel_out_dim=3,
                 ):
        super().__init__()

        self.env_num = env_num
        self.hmap_h = heightmap_decoder_out_h_dim
        self.hmap_w = heightmap_decoder_out_w_dim

        # proprioceptive
        # encoder
        self.prop_encoder = nn.Sequential(
            nn.Linear(prop_obs_in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
        )

        # exteroceptive
        # pointnet
        self.pointnet_first = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # nn.Conv1d(128, 512, kernel_size=1, bias=False),
            # # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
        )
        # self.pointnet_second = nn.Sequential(
        #     nn.Conv1d(32 + 32, 128, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(128, 256, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        # )

        # mix gru
        self.heightmap_gru = nn.GRU(
            input_size=128 + 128,
            hidden_size=128,
            num_layers=1,
            batch_first=False,
            # dropout=0.1,
        )
        self.heightmap_gru_last_h = None

        # heightmap rough decoder
        self.heightmap_rough_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, self.hmap_h * self.hmap_w),
        )
        # heightmap fine decoder
        # self.heightmap_fine_decoder = UNetRefine(in_channels=1, out_channels=1)

        # # cnn
        self.cnn = HeightmapEncoder()

        # transformer
        # self.type_embedding = nn.Embedding(2, 128)

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=128,
        #     nhead=1,
        #     dim_feedforward=256,
        #     dropout=0.1,
        #     activation="gelu",
        #     batch_first=True,
        # )
        # self.transformer = nn.TransformerEncoder(
        #     encoder_layer,
        #     num_layers=2,
        # )

        mixer_block = MixerBlock(
            num_tokens=56,
            hidden_dim=128,
            token_mlp_dim=256,
            channel_mlp_dim=256,
        )
        # 堆叠 2 层
        self.transformer = nn.Sequential(
            mixer_block,
            # mixer_block,  # 如果想让每层参数独立，可以改成 nn.ModuleList([...]) 并各自 new 一个 block
        )

        self.tf_out_proj_prop = nn.Linear(128, 128)
        self.tf_out_proj_map = nn.Linear(128 * 55, 256)

        # # gru head
        self.obs_mean_latent = nn.Linear(128, obs_latent_out_dim)
        self.obs_logvar_latent = nn.Linear(128, obs_latent_out_dim)
        self.heightmap_latent = nn.Linear(256, heightmap_latent_out_dim)
        self.footheight_latent = nn.Linear(256, footheight_latent_out_dim)

        self.head_vel = nn.Linear(128, vel_out_dim)

        # # decoder
        self.prop_obs_decoder = nn.Sequential(
            nn.Linear(obs_latent_out_dim, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, prop_decoder_out_dim),
        )

        self.heightmap_decoder = nn.Sequential(
            nn.Linear(heightmap_latent_out_dim, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, self.hmap_h * self.hmap_w),
        )

        self.footheight_decoder = nn.Sequential(
            nn.Linear(footheight_latent_out_dim, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, footheight_decoder_out_dim),
        )

    def reset_state(self):
        """
        reset the hidden state of the GRU
        """
        self.heightmap_gru_last_h = None

    def reset_state_dones(self, dones: torch.Tensor):
        """
        reset the hidden state of the GRU
        """
        if self.heightmap_gru_last_h is not None:
            self.heightmap_gru_last_h[:, dones, :] = 0

    def get_heightmap_gru_last_h(self):
        return self.heightmap_gru_last_h

    def forward(self):
        raise NotImplementedError

    def cenet_forward(self,
                      prop_history: torch.Tensor,  # (T,K,prop_obs_in_dim)
                      point_history: torch.Tensor,  # (T,K,point_num*3)
                      heightmap_gru_hidden_states: torch.Tensor | None = None,  # (T,K,heightmap_gru_hidden)
                      masks: torch.Tensor | None = None,  # (T,K)
                      p_boot_mean: float = 1.0,
                      heightmap_gt: torch.Tensor | None = None,  # (T,B,1525)
                      deterministic: bool = False,
                      use_ground_truth: bool = False
                      ):

        T = prop_history.size(0)  # time sequence
        K = prop_history.size(1)  # trajectory

        point_history = point_history.reshape(T, K, -1, 3)
        Q = point_history.size(2)  # point_num

        if heightmap_gru_hidden_states is not None:
            self.heightmap_gru_last_h = heightmap_gru_hidden_states.clone().detach()
        elif self.heightmap_gru_last_h is None or self.heightmap_gru_last_h.size(1) != K:
            self.heightmap_gru_last_h = torch.zeros(self.heightmap_gru.num_layers, K, self.heightmap_gru.hidden_size,
                                                    device=next(self.parameters()).device)

        if masks is None:
            masks = torch.ones(T, K, dtype=torch.bool, device=prop_history.device)

        # === 1) Proprioceptive  ===
        # 1.1 MLP encode each frame -> (T, K, z_t_prop_dim)
        prop_history_unpad = unpad_trajectories(prop_history, masks)  # -> (T, B, prop_obs_dim * prop_obs_his)
        z_t_prop = self.prop_encoder(prop_history)  # -> (T, B, prop_obs_out_dim)
        # print(f"z_t_prop: {z_t_prop.shape}")
        z_t_prop_unpad = unpad_trajectories(z_t_prop, masks)
        z_t_prop_unpad = z_t_prop_unpad.reshape(-1, z_t_prop_unpad.size(-1))  # (T*B, prop_obs_out_dim)
        # type_idx_prop = torch.zeros(1, dtype=torch.long, device=z_t_prop_unpad.device)  # (1,)
        # type_emb_prop = self.type_embedding(type_idx_prop)                     # (1, d_model)
        # z_t_prop_unpad = z_t_prop_unpad + type_emb_prop.view(1, -1)
        # z_t_prop_repad = self.repad_trajectories(z_t_prop_unpad, masks)  # -> (T, K, prop_obs_out_dim)

        # === 2) Exteroceptive  ===
        if not use_ground_truth:
            # sq_norm = point_history.pow(2).sum(dim=-1)      # -> (T, K, Q)
            # mask_point = sq_norm >= 1e-3              # -> (T, K, Q)

            # masks_unsq = masks.unsqueeze(-1)               # -> (T, K, 1)
            # combined_mask_tb = mask_point & masks_unsq      # -> (T, K, Q)

            point_history_flat = point_history.reshape(T * K, Q, 3).permute(0, 2, 1)  # -> (T*K, 3, Q)
            # combined_mask_flat = combined_mask_tb.reshape(T * K, Q)      # -> (T*K, Q)
            # combined_mask_unsq = combined_mask_flat.unsqueeze(1)         # -> (T*K, 1, Q)

            # first pointnet
            out_first = self.pointnet_first(point_history_flat)                    # -> (T*K, 64, Q)
            # out_first = out_first.masked_fill(~combined_mask_unsq, 0.0)
            feat_first, _ = out_first.max(dim=2)  # -> (T*K, 64)
            # feat_first_expanded = feat_first.unsqueeze(2).expand(-1, -1, Q)  # -> (T*K, 64, Q)
            # feat_first_cat = torch.cat([out_first, feat_first_expanded], dim=1)  # -> (T*K, 128, Q)

            # # second pointnet
            # out_second = self.pointnet_second(feat_first_cat)  # -> (T*K, 256, Q)
            # # out_second = out_second.masked_fill(~combined_mask_unsq, 0.0)  # -> (T*K, 256, Q)
            # feat_second, _ = out_second.max(dim=2)  # -> (T*K, 256)

            # heightmap gru
            heightmap_gru_input = torch.cat([z_t_prop, feat_first.reshape(T, K, -1)], dim=-1)  # -> (T*K, 128 + 1024)
            heightmap_gru_out, heightmap_gru_new_h = self.heightmap_gru(heightmap_gru_input, self.heightmap_gru_last_h)  # -> (T, K, 128)
            self.heightmap_gru_last_h = heightmap_gru_new_h.detach()  # -> (1, K, 128)
            heightmap_gru_out = unpad_trajectories(heightmap_gru_out, masks)  # -> (T, K', 128)

            # heightmap rough/fine decoder
            heightmap_rough_decoded = self.heightmap_rough_decoder(heightmap_gru_out).reshape(-1, self.hmap_h * self.hmap_w)  # -> (T, K', h*w)
            heightmap_fine_decoded = heightmap_rough_decoded
            # heightmap_rough_decoded_reshape = heightmap_rough_decoded.reshape(-1, 1, self.hmap_h, self.hmap_w)
            # heightmap_fine_decoded = self.heightmap_fine_decoder(heightmap_rough_decoded_reshape)  # -> [T*K',1,h,w]
            # heightmap_fine_decoded = heightmap_fine_decoded.reshape(-1, self.hmap_h * self.hmap_w)

            # cnn
            # if heightmap_gt is not None:
            #     print("heightmap_fine_decoded", heightmap_fine_decoded.shape)
            #     print("heightmap_gt", heightmap_gt.shape)
            #     cnn_input = p_boot_mean * heightmap_fine_decoded + (1 - p_boot_mean) * heightmap_gt
            # else:
            #     cnn_input = heightmap_fine_decoded
            cnn_input = heightmap_fine_decoded
        else:
            heightmap_rough_decoded = heightmap_gt
            heightmap_fine_decoded = heightmap_gt
            cnn_input = heightmap_gt

        cnn_input = cnn_input.reshape(-1, 1, self.hmap_h, self.hmap_w)
        cnn_out = self.cnn(cnn_input)
        # type_idx_pt = torch.ones(1, dtype=torch.long, device=cnn_out.device)     # (1,)
        # type_emb_pt = self.type_embedding(type_idx_pt)                         # (1, d_model)
        # cnn_out = cnn_out + type_emb_pt.view(1, -1)

        # transformer
        tf_input = torch.cat([z_t_prop_unpad.unsqueeze(1), cnn_out], dim=1)
        # print(f"tf_input: {tf_input.shape}")
        tf_out = tf_input
        for block in self.transformer:
            tf_out = block(tf_out)
        out_prop = tf_out[:, 0, :]                     # (K,C)
        out_map = tf_out[:, 1:, :]                    # (K,token_num-1,C)
        # out_map_pool, _ = out_map.max(dim=1)          # (K,C)
        # out_ac = torch.cat([out_prop, out_map_pool], dim=1)  # (K,2C)
        # print(f"tf_out: {tf_out.shape}")
        out_proj_prop = self.tf_out_proj_prop(out_prop)
        out_proj_map = self.tf_out_proj_map(out_map.flatten(start_dim=1))

        # multi-head VAE branches
        mean_obs = self.obs_mean_latent(out_proj_prop)       # (T, K, L, latent_out_dim)
        logvar_obs = self.obs_logvar_latent(out_proj_prop)
        code_v = self.head_vel(out_proj_prop)
        code_fh = self.footheight_latent(out_proj_map)
        code_hmap = self.heightmap_latent(out_proj_map)

        # clamp logvar
        logvar_obs = torch.clamp(logvar_obs, min=-10, max=10)

        # reparameterise
        code_obs_latent = self.reparameterise(mean_obs, logvar_obs, deterministic)  # (T, K, L, latent)

        # concat all latent channels
        code = torch.cat([code_v, code_obs_latent, code_fh, code_hmap], dim=-1)  # (T, K, L, sum_latent)

        # decode
        prop_obs_decoded = self.prop_obs_decoder(code_obs_latent)         # (T, K, L, prop_dim)
        heightmap_decoded = self.heightmap_decoder(code_hmap)   # (T, K, L, heightmap_out_dim)
        footheight_decoded = self.footheight_decoder(code_fh)   # (T, K, L, footheight_out_dim)

        return {
            "code": code,
            "code_vel": code_v,
            "code_heightmap_latent": code_hmap,
            "code_footheight_latent": code_fh,
            "prop_obs_decoded": prop_obs_decoded,
            "heightmap_decoded": heightmap_decoded,
            "code_obs_latent": code_obs_latent,
            "mean_obs": mean_obs,
            "logvar_obs": logvar_obs,
            "heightmap_rough_decoded": heightmap_rough_decoded,
            "heightmap_fine_decoded": heightmap_fine_decoded,
            "footheight_decoded": footheight_decoded,
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
                      prop_history: torch.Tensor,
                      point_history: torch.Tensor,
                      gt_heightmap: torch.Tensor | None = None,
                      use_ground_truth: bool = False
                      ):
        dict_out = self.cenet_forward(prop_history, point_history, deterministic=True, heightmap_gt=gt_heightmap, use_ground_truth=use_ground_truth)
        return dict_out["code"], dict_out["heightmap_rough_decoded"], dict_out["heightmap_fine_decoded"], dict_out["heightmap_decoded"], dict_out["footheight_decoded"]

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

    def repad_trajectories(self, unpadded, masks, pad_value=0.0):
        T, K = masks.shape
        D = unpadded.size(-1)
        # 1) 初始化 (T, K, D)
        padded = torch.full((T, K, D),
                            fill_value=pad_value,
                            device=unpadded.device,
                            dtype=unpadded.dtype)
        # 2) 按“时间-批次”顺序扁平化
        padded_flat = padded.view(T * K, D)      # 行号 = t * K + b
        mask_flat = masks.flatten()            # 同样是行号 = t * K + b
        # unpadded 原来是 (T, K', D)，直接按 row-major flatten 得到与 mask_flat 对齐的值序列
        values = unpadded.reshape(-1, D)         # 行号 = t * K' + i
        # 3) 写回
        padded_flat[mask_flat] = values
        # 4) 恢复回 (T, K, D)
        return padded
