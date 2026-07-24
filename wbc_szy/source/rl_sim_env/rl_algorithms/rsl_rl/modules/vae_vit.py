from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class PartialConv3dSame(nn.Module):
    """
    3D 体素版 PConv（SAME padding + 边界视为洞）:
      x:    (B, C_in, D, H, W)
      mask: (B, 1,     D, H, W)  有效=1, 无效=0；None 则默认全 1

    规则：
      - 输出空间尺寸: D_out=ceil(D_in/stride_d), H_out=ceil(H_in/stride_h), W_out=ceil(W_in/stride_w)
      - SAME 的多出来 1 个体素优先分配到后/下/右（与 TF 对齐）
      - 无有效体素（卷积窗口内 mask 全 0）时，该位置输出必须为 0
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int, int],
                 stride: int | tuple[int, int, int] = 1,
                 dilation: int | tuple[int, int, int] = 1,
                 bias: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.kd, self.kh, self.kw = kernel_size
        self.sd, self.sh, self.sw = stride
        self.dd, self.dh, self.dw = dilation
        self.eps = eps

        # 主卷积设 padding=0（使用手工 SAME pad）
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, dilation=dilation, bias=bias)

        # 掩码计数核：全 1，形状 (1,1,kd,kh,kw)
        self.register_buffer('mask_kernel', torch.ones(1, 1, self.kd, self.kh, self.kw))
        self.window_size = self.kd * self.kh * self.kw

    @staticmethod
    def _same_pad_1d(in_size: int, k: int, s: int, d: int):
        eff_k = (k - 1) * d + 1
        out_size = math.ceil(in_size / s)
        pad_needed = max(0, (out_size - 1) * s + eff_k - in_size)
        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before   # 额外 1 给 "after"（后/下/右）
        return pad_before, pad_after

    def _pad_xm(self, x: torch.Tensor, m: torch.Tensor | None):
        D, H, W = x.shape[-3:]
        pfd, pad = self._same_pad_1d(D, self.kd, self.sd, self.dd)
        pfh, pah = self._same_pad_1d(H, self.kh, self.sh, self.dh)
        pfw, paw = self._same_pad_1d(W, self.kw, self.sw, self.dw)
        # F.pad 3D 的顺序: (wL, wR, hT, hB, dF, dB)  ← 注意维度顺序
        pad_tuple = (pfw, paw, pfh, pah, pfd, pad)
        x_p = F.pad(x, pad_tuple, mode='constant', value=0.0)
        if m is None:
            m = torch.ones_like(x[:, :1, :, :, :])
        m_p = F.pad(m, pad_tuple, mode='constant', value=0.0)  # 边界外 mask=0
        return x_p, m_p

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # 1) 数值清理，避免 NaN/Inf 传播
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) 严格 SAME 手工填充（后/下/右优先）
        x_p, m_p = self._pad_xm(x, mask)

        # 3) 有效体素计数（与主卷积同 stride/dilation，padding=0）
        with torch.no_grad():
            mask_sum = F.conv3d(m_p, self.mask_kernel,
                                stride=(self.sd, self.sh, self.sw),
                                padding=0,
                                dilation=(self.dd, self.dh, self.dw))
            new_mask = (mask_sum > 0).float()
            mask_ratio = (self.window_size / (mask_sum + self.eps)) * new_mask

        # 4) 卷积 + 局部归一化（并强制 new_mask 置零无效位置）
        y = self.conv(x_p * m_p)
        if self.conv.bias is not None:
            b = self.conv.bias.reshape(1, -1, 1, 1, 1)
            y = (y - b) * mask_ratio + b
        else:
            y = y * mask_ratio
        y = y * new_mask  # 无有效体素 → 输出 0
        return y, new_mask


class PartialConv2dSame(nn.Module):
    """
    严格 SAME padding 的 PConv：
      - SAME 计算含 stride/dilation，一致于 TensorFlow：
          out = ceil(in / stride)
          pad_total = max(0, (out-1)*stride + eff_kernel - in)
          eff_kernel = (k-1)*dilation + 1
        剩余 1 个像素的 pad 分配给右/下侧（与 TF SAME 对齐）。
      - 边界之外一律视为洞（mask=0）。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kh, self.kw = kernel_size
        self.sh, self.sw = stride
        self.dh, self.dw = dilation
        self.eps = eps

        # 卷积本体使用 padding=0（我们手工 SAME pad 到右/下）
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, dilation=dilation, bias=bias)

        # 掩码计数卷积核
        self.register_buffer('mask_kernel', torch.ones(1, 1, self.kh, self.kw))
        self.window_size = self.kh * self.kw

    @staticmethod
    def _same_pad(in_size: int, k: int, s: int, d: int):
        """返回 (pad_before, pad_after), 使得输出尺寸 = ceil(in_size / s)"""
        eff_k = (k - 1) * d + 1
        out_size = math.ceil(in_size / s)
        pad_needed = max(0, (out_size - 1) * s + eff_k - in_size)
        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before  # 多出来的 1 像素给 after（右/下）
        return pad_before, pad_after

    def _pad_xy(self, x: torch.Tensor, m: torch.Tensor | None):
        H, W = x.shape[-2:]
        pt, pb = self._same_pad(H, self.kh, self.sh, self.dh)
        pl, pr = self._same_pad(W, self.kw, self.sw, self.dw)
        # F.pad 的顺序为 (left, right, top, bottom)
        pad = (pl, pr, pt, pb)
        x_p = F.pad(x, pad, mode='constant', value=0.0)
        if m is None:
            m = torch.ones_like(x[:, :1, :, :])
        m_p = F.pad(m, pad, mode='constant', value=0.0)  # 边界外 mask=0
        return x_p, m_p

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # 1) 数值清理
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) 严格 SAME 手工填充（右/下优先）
        x_p, m_p = self._pad_xy(x, mask)

        # 3) 有效像素计数（与主卷积同 stride/dilation，padding=0）
        with torch.no_grad():
            mask_sum = F.conv2d(m_p, self.mask_kernel,
                                stride=(self.sh, self.sw),
                                padding=0,
                                dilation=(self.dh, self.dw))
            new_mask = (mask_sum > 0).float()
            mask_ratio = (self.window_size / (mask_sum + self.eps)) * new_mask

        # 4) 卷积 + 局部归一化（最后强制乘 new_mask，保证“无有效像素→输出为 0”）
        y = self.conv(x_p * m_p)
        if self.conv.bias is not None:
            b = self.conv.bias.reshape(1, -1, 1, 1)
            y = (y - b) * mask_ratio + b
        else:
            y = y * mask_ratio
        y = y * new_mask
        return y, new_mask


class PartialConv2d(nn.Module):
    """
    共享单通道掩码版 PConv：
      x:    (B, C_in, H, W)
      mask: (B, 1,    H, W)  有效=1, 无效=0；None 则默认全 1
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, bias=bias)
        self.register_buffer('mask_kernel', torch.ones(1, 1, kernel_size, kernel_size))
        self.window_size = kernel_size * kernel_size
        self.eps = 1e-6

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # 清理 NaN/Inf，避免污染
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 1) 掩码初始化
        if mask is None:
            mask = torch.ones_like(x[:, :1, :, :])  # (B,1,H,W)

        # 2) 有效像素计数（随 stride/pad/dilation 下采样）
        with torch.no_grad():
            mask_sum = F.conv2d(mask, self.mask_kernel,
                                stride=self.conv.stride,
                                padding=self.conv.padding,
                                dilation=self.conv.dilation)
            mask_ratio = (self.window_size / (mask_sum + self.eps)) * (mask_sum > 0).float()
            new_mask = (mask_sum > 0).float()

        # 3) 卷积 + 归一化
        y = self.conv(x * mask)
        if self.conv.bias is not None:
            b = self.conv.bias.reshape(1, -1, 1, 1)
            y = (y - b) * mask_ratio + b
        else:
            y = y * mask_ratio

        return y, new_mask


class MultiLayerPConvNet(nn.Module):
    def __init__(self, in_ch: int, layer_params: list[dict]):
        super().__init__()
        self.pconvs = nn.ModuleList()
        self.gns = nn.ModuleList()  # 只给中间层配 GN
        ch_in = in_ch
        for i, cfg in enumerate(layer_params):
            ch_out = cfg['out_ch']
            self.pconvs.append(PartialConv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=cfg['kernel_size'],
                stride=cfg.get('stride', 1),
                padding=cfg.get('padding', 0),
                dilation=cfg.get('dilation', 1),
                bias=cfg.get('bias', True),
            ))
            ch_in = ch_out
        # 给除最后一层外的每层准备 GN，groups 选能整除通道数的“<=8 的最大因子”
        for i, cfg in enumerate(layer_params[:-1]):
            c = cfg['out_ch']
            g = 8
            while c % g != 0 and g > 1:
                g -= 1
            self.gns.append(nn.GroupNorm(num_groups=g, num_channels=c))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None):
        feat, m = x, mask
        # 中间层：PConv -> GN -> ELU
        for pconv, gn in zip(self.pconvs[:-1], self.gns):
            feat, m = pconv(feat, m)
            feat = gn(feat)
            feat = F.elu(feat, inplace=True)
        # 最后一层：只做 PConv（不归一化不激活）
        feat, m = self.pconvs[-1](feat, m)
        return feat, m


class HeightmapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # (61, 25) -> (11, 5)
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (11, 5) -> (11, 5)
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (11, 5) -> (11, 5)
            # nn.ELU(inplace=True),
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
        x = x.reshape(K, C, H * W)
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
                 point_history_in_dim=2,
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
        self.point_history_in_dim = point_history_in_dim

        # proprioceptive
        # encoder
        self.prop_encoder = nn.Sequential(
            nn.Linear(prop_obs_in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
        )

        # exteroceptive
        # partial heightmap cnn
        layers = [
            {'out_ch': 32, 'kernel_size': 7, 'stride': 4, 'padding': 3},
            {'out_ch': 64, 'kernel_size': 5, 'stride': 2, 'padding': 2},
            {'out_ch': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_ch': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        ]
        self.partial_hmap_cnn = MultiLayerPConvNet(in_ch=1, layer_params=layers)

        partial_hmap_mixer_block = MixerBlock(
            num_tokens=(self.point_history_in_dim * 32 + 1),
            hidden_dim=128,
            token_mlp_dim=256,
            channel_mlp_dim=256,
        )
        # 堆叠 2 层
        self.partial_hmap_mixer = nn.Sequential(
            partial_hmap_mixer_block,
            # mixer_block,  # 如果想让每层参数独立，可以改成 nn.ModuleList([...]) 并各自 new 一个 block
        )

        # self.partial_hmap_out_proj_prop = nn.Linear(128, 128)
        # self.partial_hmap_out_proj_map = nn.Linear(128 * 42 * 4, 128 * 4)
        self.partial_hmap_out_proj = nn.Linear(128 * (self.point_history_in_dim * 32 + 1), 128)
        # mix gru
        self.heightmap_gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=False,
            # dropout=0.1,
        )
        self.heightmap_gru_last_h = None

        # heightmap rough decoder
        self.heightmap_rough_decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, self.hmap_h * self.hmap_w),
        )
        # heightmap fine decoder
        self.heightmap_fine_decoder = UNetRefine(in_channels=1, out_channels=1)

        # # cnn
        self.cnn_full = HeightmapEncoder()

        mixer_block = MixerBlock(
            num_tokens=33,
            hidden_dim=128,
            token_mlp_dim=256,
            channel_mlp_dim=256,
        )
        # 堆叠 2 层
        self.mlp_mixer = nn.Sequential(
            mixer_block,
            # mixer_block,  # 如果想让每层参数独立，可以改成 nn.ModuleList([...]) 并各自 new 一个 block
        )

        self.out_proj_prop = nn.Linear(128, 128)
        self.out_proj_map = nn.Linear(128 * 32, 256)

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
                      point_history: torch.Tensor,  # (T,K,his*num*partial_hmap_obs_num)
                      heightmap_gru_hidden_states: torch.Tensor | None = None,  # (T,K,heightmap_gru_hidden)
                      masks: torch.Tensor | None = None,  # (T,K)
                      p_boot_mean: float = 1.0,
                      heightmap_gt: torch.Tensor | None = None,  # (T,B,1525)
                      deterministic: bool = False,
                      use_ground_truth: bool = False,
                      use_adaboot: bool = False
                      ):

        T = prop_history.size(0)  # time sequence
        K = prop_history.size(1)  # trajectory

        if masks is None:
            masks = torch.ones(T, K, dtype=torch.bool, device=prop_history.device)

        if heightmap_gru_hidden_states is not None:
            self.heightmap_gru_last_h = heightmap_gru_hidden_states.clone().detach()
        elif self.heightmap_gru_last_h is None or self.heightmap_gru_last_h.size(1) != K:
            self.heightmap_gru_last_h = torch.zeros(self.heightmap_gru.num_layers, K, self.heightmap_gru.hidden_size,
                                                    device=next(self.parameters()).device)

        # === 1) Proprioceptive  ===
        # 1.1 MLP encode each frame -> (T, K, z_t_prop_dim)
        z_t_prop = self.prop_encoder(prop_history)  # -> (T, B, prop_obs_out_dim)
        # print(f"z_t_prop: {z_t_prop.shape}")
        z_t_prop_unpad = unpad_trajectories(z_t_prop, masks)
        z_t_prop_unpad = z_t_prop_unpad.reshape(-1, z_t_prop_unpad.size(-1))  # (T*B, prop_obs_out_dim)

        # === 2) Exteroceptive  ===
        if not use_ground_truth:
            C = self.point_history_in_dim
            H = self.hmap_h
            W = self.hmap_w

            # (T,K, C*H*W) -> (T,K,C,H,W)
            point_history = point_history.reshape(T, K, C, H, W)

            # 像素级 mask（你已确认无 inf，阈值 -10 即可）
            point_mask = (point_history > -10)

            # 轨迹级 (T,K) -> (T,K,C,H,W)，做交集
            tk_mask_px = masks[..., None, None, None].expand(T, K, C, H, W)
            point_mask = point_mask & tk_mask_px

            # reshape 给 PConv
            B = T * K * C
            point_history = point_history.reshape(B, 1, H, W)
            point_mask = point_mask.reshape(B, 1, H, W).float()

            # PConv 编码 -> (T*K*4, 128, 1, 1)
            feat, _p_mask = self.partial_hmap_cnn(point_history, point_mask)

            # 把每个 (t,k) 的 4 个 patch 展成一条 512 维向量
            feat = feat.reshape(T, K, self.point_history_in_dim * 32, -1)  # (T, K, 512)
            z_t_prop = z_t_prop.reshape(T, K, 1, -1)
            # === GRU 输入：128(本体) + 512(局部) = 640，与构造的 GRU input_size 匹配 ===
            mixer_input = torch.cat([z_t_prop, feat], dim=2)
            mixer_input = mixer_input.reshape(T * K, self.point_history_in_dim * 32 + 1, -1)
            mixer_out = mixer_input
            for block_hmap in self.partial_hmap_mixer:
                mixer_out = block_hmap(mixer_out)
            heightmap_gru_input = self.partial_hmap_out_proj(mixer_out.reshape(T * K, -1))  # (T, K, 640)
            heightmap_gru_input = heightmap_gru_input.reshape(T, K, -1)
            heightmap_gru_out, heightmap_gru_new_h = self.heightmap_gru(heightmap_gru_input, self.heightmap_gru_last_h)
            self.heightmap_gru_last_h = heightmap_gru_new_h.detach()
            heightmap_gru_out = unpad_trajectories(heightmap_gru_out, masks)  # -> (T, K', 128)

            # heightmap rough/fine
            heightmap_rough_decoded = self.heightmap_rough_decoder(heightmap_gru_out).reshape(-1, self.hmap_h * self.hmap_w)
            heightmap_rough_decoded_reshape = heightmap_rough_decoded.reshape(-1, 1, self.hmap_h, self.hmap_w)
            heightmap_fine_decoded = self.heightmap_fine_decoder(heightmap_rough_decoded_reshape).reshape(-1, self.hmap_h * self.hmap_w)
            if use_adaboot:
                batch_num, _ = heightmap_gt.shape
                replace_num = int(batch_num * (1 - p_boot_mean))
                if replace_num > 0:
                    row_idx = torch.randperm(batch_num)[:replace_num]
                    heightmap_fine_decoded[row_idx] = heightmap_gt[row_idx]

            cnn_input = heightmap_fine_decoded
        else:
            heightmap_rough_decoded = heightmap_gt
            heightmap_fine_decoded = heightmap_gt
            cnn_input = heightmap_gt

        cnn_input = cnn_input.reshape(-1, 1, self.hmap_h, self.hmap_w)
        cnn_out = self.cnn_full(cnn_input)
        # type_idx_pt = torch.ones(1, dtype=torch.long, device=cnn_out.device)     # (1,)
        # type_emb_pt = self.type_embedding(type_idx_pt)                         # (1, d_model)
        # cnn_out = cnn_out + type_emb_pt.reshape(1, -1)

        # transformer
        tf_input = torch.cat([z_t_prop_unpad.unsqueeze(1), cnn_out], dim=1)
        # print(f"tf_input: {tf_input.shape}")
        tf_out = tf_input
        for block in self.mlp_mixer:
            tf_out = block(tf_out)
        out_prop = tf_out[:, 0, :]                     # (K,C)
        out_map = tf_out[:, 1:, :]                    # (K,token_num-1,C)
        # out_map_pool, _ = out_map.max(dim=1)          # (K,C)
        # out_ac = torch.cat([out_prop, out_map_pool], dim=1)  # (K,2C)
        # print(f"tf_out: {tf_out.shape}")
        out_proj_prop = self.out_proj_prop(out_prop)
        out_proj_map = self.out_proj_map(out_map.flatten(start_dim=1))

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

    def reparameterise(self, mean: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False):
        if deterministic:
            return mean
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps

    @torch.no_grad()
    def cenet_forward_export(self,
                             prop_t: torch.Tensor,    # (K, prop_obs_in_dim)
                             point_t: torch.Tensor,   # (K, C*H*W) , C=self.point_history_in_dim
                             h_prev: torch.Tensor     # (num_layers=1, K, 128)
                             ):
        self.eval()
        T = 1
        K = prop_t.size(0)
        C, H, W = self.point_history_in_dim, self.hmap_h, self.hmap_w

        # === 1) Proprio ===
        prop_history = prop_t.unsqueeze(0)                # (1, K, Dp)
        z_t_prop = self.prop_encoder(prop_history)        # (1, K, 128)
        # 导出时不做 unpad，直接摊平到 (K, 128)
        z_t_prop_unpad = z_t_prop.reshape(-1, z_t_prop.size(-1))   # (K, 128)

        # === 2) Extero ===
        ph = point_t.unsqueeze(0).reshape(T, K, C, H, W)  # (1,K,C,H,W)
        point_mask = (ph > -10).to(ph.dtype)              # 元素级比较 → float
        B = T * K * C
        ph_b = ph.reshape(B, 1, H, W)                     # (B,1,H,W)
        pm_b = point_mask.reshape(B, 1, H, W)             # (B,1,H,W)

        feat, _ = self.partial_hmap_cnn(ph_b, pm_b)       # (B, 128, h', w')
        # 与原实现保持一致的 reshape 方式（不要改动 token 维度逻辑）
        feat = feat.reshape(T, K, self.point_history_in_dim * 32, -1)  # (1,K, 32*C, d)

        z_prop_tok = z_t_prop.reshape(T, K, 1, -1)        # (1,K,1,128)
        mixer_input = torch.cat([z_prop_tok, feat], dim=2)            # (1,K, 32*C+1, 128)
        mixer_input = mixer_input.reshape(T * K, self.point_history_in_dim * 32 + 1, -1)
        mixer_out = mixer_input
        for block_hmap in self.partial_hmap_mixer:
            mixer_out = block_hmap(mixer_out)             # (K, tokens, 128)

        hmap_gru_in = self.partial_hmap_out_proj(mixer_out.reshape(T * K, -1))  # (K,128)
        hmap_gru_in = hmap_gru_in.reshape(T, K, -1)       # (1,K,128)
        hmap_gru_out, h_new = self.heightmap_gru(hmap_gru_in, h_prev)         # (1,K,128), (1,K,128)

        # 不做 unpad；直接摊平成 (K,128)
        hmap_gru_out_2d = hmap_gru_out.reshape(T * K, -1)   # (K,128)

        # === 3) 粗/细高程图 ===
        hm_rough = self.heightmap_rough_decoder(hmap_gru_out_2d)              # (K, H*W)
        hm_rough_img = hm_rough.reshape(-1, 1, H, W)                           # (K,1,H,W)
        hm_fine_img = self.heightmap_fine_decoder(hm_rough_img)                # (K,1,H,W)
        hm_fine = hm_fine_img.reshape(-1, H * W)                                  # (K,H*W)

        # === 4) CNN + MLP-Mixer 主干 ===
        cnn_in = hm_fine_img                                                    # (K,1,H,W)
        cnn_tok = self.cnn_full(cnn_in)                                         # (K, N_tok, 128)

        tf_input = torch.cat([z_t_prop_unpad.unsqueeze(1), cnn_tok], dim=1)     # (K, 1+N_tok, 128)
        tf_out = tf_input
        for block in self.mlp_mixer:
            tf_out = block(tf_out)                                              # (K, 1+N_tok, 128)

        out_prop = tf_out[:, 0, :]                                              # (K,128)
        out_map = tf_out[:, 1:, :]                                             # (K,N_tok,128)

        out_proj_prop = self.out_proj_prop(out_prop)                             # (K,128)
        out_proj_map = self.out_proj_map(out_map.flatten(start_dim=1))          # (K,256)

        # === 5) 多头 VAE（导出：确定性，不采样） ===
        mean_obs = self.obs_mean_latent(out_proj_prop)                          # (K, Lobs)
        code_obs_latent = mean_obs                                               # deterministic

        code_v = self.head_vel(out_proj_prop)                                 # (K,3)
        code_fh = self.footheight_latent(out_proj_map)                         # (K, Lfh)
        code_hmap = self.heightmap_latent(out_proj_map)                          # (K, Lhmap)

        code = torch.cat([code_v, code_obs_latent, code_fh, code_hmap], dim=-1)  # (K, sum_latent)

        return (code, h_new, hm_fine)

    def act_inference(self,
                      prop_history: torch.Tensor,
                      point_history: torch.Tensor,
                      gt_heightmap: torch.Tensor | None = None,
                      use_ground_truth: bool = False,
                      h_prev: torch.Tensor | None = None
                      ):
        dict_out = self.cenet_forward(prop_history, point_history, heightmap_gru_hidden_states=h_prev, deterministic=True, heightmap_gt=gt_heightmap, use_ground_truth=use_ground_truth)
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
