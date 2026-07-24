import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points, ball_query
from pytorch3d.ops.ball_query import _ball_query, _KNN, masked_gather


# -----------------------
# 置信度过滤模块
# -----------------------
class ConfidenceFilter(nn.Module):
    """
    C_filtered = C * (1 - tanh(std(C, dim=-1)))
    输入 feat: (N, C, L)，输出同形状 (N, C, L)
    """

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (N, C, L)
        sigma = feat.std(dim=2, unbiased=False)     # (N, C)
        mask = 1.0 - torch.tanh(sigma)               # (N, C)
        return feat * mask.unsqueeze(-1)             # (N, C, L)


# -----------------------
# 带有效点标记的 FPS + QueryBall
# -----------------------
def sample_farthest_points_history(xyz: torch.Tensor,
                                   n: int,
                                   h: int,
                                   m: int,
                                   num_samples: int):
    """
    对包含历史信息的点云数据进行最远点采样，并记录每个点的时间步 ID，返回原始索引和时间步 ID。

    参数：
        xyz (Tensor): 输入点云数据，形状为 (n, h * m, 3)
        n (int): 环境数
        h (int): 历史步数
        m (int): 每步的点数
        num_samples (int): 每个时间步需要采样的点数

    返回：
        sampled_indices (Tensor): 每个时间步的采样点的原始索引和时间步 ID，形状为 (n, h, num_samples, 4)
    """
    # 将点云数据重塑为 (n, h, m, 3)，恢复每个环境的历史步数据
    xyz_reshaped = xyz.reshape(n, h, m, 3)

    # 将数据重塑为 (n * h, m, 3)，以便批量处理所有时间步
    xyz_flat = xyz_reshaped.reshape(n * h, m, 3)

    # 批量进行最远点采样
    _, sampled_indices_flat = sample_farthest_points(xyz_flat, K=num_samples, random_start_point=True)

    # 创建时间步 ID，形状为 (n * h, num_samples, 1)
    time_step_id = torch.arange(h, device=xyz.device).repeat(n, 1).view(-1, 1)  # shape (n * h, 1)
    time_step_id = time_step_id.unsqueeze(1).expand(-1, num_samples, -1)  # shape (n * h, num_samples, 1)

    # 将原始索引和时间步 ID 合并，形状为 (n * h, num_samples, 4)
    sampled_indices_with_time_step = torch.cat([sampled_indices_flat.unsqueeze(-1), time_step_id.float()], dim=-1)

    # 重塑为 (n, h, num_samples, 4)
    sampled_indices_with_time_step_all_steps = sampled_indices_with_time_step.reshape(n, h * num_samples, 4)

    return sampled_indices_with_time_step_all_steps


def grouping_layer_using_ball_query(xyz: torch.Tensor,
                                    n: int,
                                    h: int,
                                    m: int,
                                    num_samples: int,
                                    K: int,
                                    radius: float):
    """
    使用Ball Query来为每个采样点从其局部区域中挑选K个领域点。

    参数：
        xyz (Tensor): 原始点云数据，形状为 (n, N, 3)，
                       其中 `n` 是环境数，`N` 是原始点云中的点数，`3` 是坐标维度。
        sampled_indices_with_time_step_all_steps (Tensor): 采样后的索引和时间步 ID，形状为 (n, h * num_samples, 4)。
        n (int): 环境数（batch size）。
        h (int): 历史步数（时间步数）。
        num_samples (int): 每个时间步采样的点数。
        K (int): 每个采样点选择的邻居点数量。
        radius (float): 搜索半径。

    返回：
        grouped_points (Tensor): 每个采样点的邻域点，形状为 (n, h * num_samples, K, 3)。
    """
    sampled_indices_with_time_step_all_steps = sample_farthest_points_history(xyz, n, h, m, num_samples)
    # 获取采样点的坐标，形状为 (n, h * num_samples, 3)
    sampled_points = sampled_indices_with_time_step_all_steps[..., :3]

    # Ball Query 通过球形区域来查找邻居点
    grouped_points = ball_query(p1=sampled_points, p2=xyz, K=K, radius=radius)

    # 返回每个采样点的 K 个邻域点
    return grouped_points.nn  # 返回邻居点的坐标 (n, h * num_samples, K, 3)

# -----------------------
# 单尺度 Set Abstraction 模块（Query Ball 版）
# -----------------------


class PartialSAModuleP3DQueryBall(nn.Module):
    """
    输入 xyzv: (N, M, 4)，其中 [:,:,:3] 是坐标，[...,3] 是有效性标记
    """

    def __init__(self, npoint: int, nsample: int, mlp_channels: list[int], radius: float):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius

        layers = []
        last_c = 3
        for out_c in mlp_channels:
            layers += [
                nn.Conv2d(last_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ]
            last_c = out_c
        self.mlp = nn.Sequential(*layers)
        self.conf = ConfidenceFilter()

    def forward(
        self,
        xyzv: torch.Tensor,         # (N, M, 4)
        idx: torch.Tensor = None,   # (N, npoint)
        new_xyz: torch.Tensor = None  # (N, npoint, 3)
    ):
        # 拆分坐标与掩码
        xyz = xyzv[..., :3]         # (N, M, 3)
        mask = xyzv[..., 3] > 0.5    # (N, M)
        N, M, _ = xyz.shape

        # 如果没给出 idx/new_xyz，就自己做一次 FPS
        if idx is None or new_xyz is None:
            _, idx = sample_farthest_from_pts4(
                xyzv, K=self.npoint, random_start_point=False
            )
            new_xyz = xyz.gather(
                1, idx.unsqueeze(-1).expand(-1, -1, 3)
            )  # (N, npoint, 3)

        # 2. Ball Query（限 K=self.nsample, 半径=self.radius）
        group_idx = ball_query(
            new_xyz, xyz,
            K=self.nsample,
            radius=self.radius,
            return_nn=False
        ).idx  # (N, npoint, nsample)

        # 3. 收集邻域 & 中心化
        grouped = xyz.unsqueeze(1).expand(-1, self.npoint, -1, 3)
        grouped = grouped.gather(
            2, group_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # (N, npoint, nsample, 3)
        grouped = grouped - new_xyz.unsqueeze(2)

        # 4. Shared-MLP
        x = grouped.permute(0, 3, 1, 2)  # → (N, 3, npoint, nsample)
        x = self.mlp(x)                  # → (N, C_out, npoint, nsample)

        # 5. ConfidenceFilter + MaxPool
        sigma = x.std(dim=-1, unbiased=False)     # (N, C_out, npoint)
        mask_conf = 1.0 - torch.tanh(sigma)       # (N, C_out, npoint)
        x = x * mask_conf.unsqueeze(-1)           # (N, C_out, npoint, nsample)
        new_feat = x.max(dim=3)[0]                # (N, C_out, npoint)

        return new_xyz, new_feat


# -----------------------
# 多尺度 Set Abstraction (MSG)
# -----------------------
class SAModuleP3D_MSG(nn.Module):
    """
    输入 xyzv: (N, M, 4)，输出 new_xyz: (N, npoint, 3)、tokens: (N, npoint, d_model)
    """

    def __init__(self, npoint: int, radii: list[float], nsamples: list[int],
                 mlp_channels: list[list[int]], d_model: int):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlp_channels)
        self.modules = nn.ModuleList([
            PartialSAModuleP3DQueryBall(npoint, k, ch, r)
            for r, k, ch in zip(radii, nsamples, mlp_channels)
        ])
        last_C = sum(ch[-1] for ch in mlp_channels)
        self.to_dmodel = nn.Conv1d(last_C, d_model, kernel_size=1)

    def forward(self, xyzv: torch.Tensor):
        # 1. 全局 FPS（统一采样）
        pts4 = xyzv
        _, idx = sample_farthest_from_pts4(
            pts4, K=self.modules[0].npoint, random_start_point=False
        )  # (N, npoint)
        xyz = xyzv[..., :3]
        new_xyz = xyz.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # (N, npoint, 3)

        # 2. 多尺度特征
        feats = []
        for m in self.modules:
            _, feat = m(xyzv, idx=idx, new_xyz=new_xyz)
            feats.append(feat)  # (N, C_i, npoint)

        # 3. concat + 映射到 d_model
        feat_cat = torch.cat(feats, dim=1)     # (N, ΣC_i, npoint)
        tokens = self.to_dmodel(feat_cat)      # (N, d_model, npoint)
        tokens = tokens.permute(0, 2, 1)       # (N, npoint, d_model)

        return new_xyz, tokens

# -----------------------
# 整体 Pipeline：PointNet2 → Transformer Token
# -----------------------


class PointNet2FeatureExtractorP3D(nn.Module):
    """
    从 (N, M, 3) 点云到 Transformer token 序列 (N, M_L, d_model)
    这里在第一层把 pts→拼有效性标记→xyzv
    """

    def __init__(self, sa_cfgs: list[dict], d_model: int):
        super().__init__()
        self.sa_layers = nn.ModuleList([
            SAModuleP3D_MSG(cfg['npoint'], cfg['radii'],
                            cfg['nsamples'], cfg['mlp_channels'],
                            d_model)
            for cfg in sa_cfgs
        ])

    def forward(self, xyzv: torch.Tensor):
        """
        输入:
            xyzv: Tensor[N, M, 4]，其中 [:,:,:3] 是点坐标，[...,3] 是有效性标记 (0/1)
        返回:
            tokens: Tensor[N, M_L, d_model]
        """
        N, M, _ = xyzv.shape

        for layer in self.sa_layers:
            new_xyz, tokens = layer(xyzv)
            # 为下一层构造新的 xyzv，全标记为有效
            Np = new_xyz.shape[1]
            xyzv = torch.cat([
                new_xyz,
                torch.ones(N, Np, 1, device=new_xyz.device)
            ], dim=-1)  # (N, Np, 4)

        return tokens  # (N, M_L, d_model)
