import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points, knn_points, ball_query

# -----------------------
# 置信度过滤模块
# -----------------------


class ConfidenceFilter(nn.Module):
    """
    C = feat * (1 - tanh(std(feat)))
    输入 feat: (N, C, L)，输出同形状 (N, C, L)
    """

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (N, C, L)
        # 计算各通道在 token 维度上的标准差
        sigma = feat.std(dim=2, unbiased=False)        # (N, C)
        # 构建 mask：低方差→接近1，高方差→接近0
        mask = 1.0 - torch.tanh(sigma)                 # (N, C)
        # 直接按论文公式逐通道乘以 mask
        return feat * mask.unsqueeze(-1)               # (N, C, L)

# -----------------------
# PointNet++ Set Abstraction
# -----------------------


def sample_farthest_from_pts4(
    pts4: torch.Tensor,        # (N, M, 4)
    K: int,
    random_start_point: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    pts = pts4[..., :3]                       # (N, M, 3)
    mask = pts4[..., 3] > 0.5                 # (N, M) bool

    N, M, D = pts.shape
    device = pts.device

    # —— 向量化重排索引 —— #
    # 我们希望 valid (mask=True) 在前，invalid 在后。
    # ~mask 会把 valid->False(0), invalid->True(1)，
    # argsort 按照从小到大排序，于是 valid 的索引自然排在前面。
    reorder_idx = (~mask).argsort(dim=1)      # (N, M) long

    # 下面保持原来逻辑
    pts_r = pts.gather(
        1,
        reorder_idx.unsqueeze(-1).expand(-1, -1, D)
    )  # (N, M, 3)

    lengths = mask.sum(dim=1).clamp(min=0, max=M).to(torch.int64).to(device)  # (N,)

    sampled_pts_r, sampled_idx_r = sample_farthest_points(
        pts_r,
        lengths=lengths,
        K=K,
        random_start_point=random_start_point
    )
    orig_idx = reorder_idx.gather(1, sampled_idx_r)  # (N, K)

    sampled_pts = pts.gather(
        1,
        orig_idx.unsqueeze(-1).expand(-1, -1, D)
    )  # (N, K, 3)

    return sampled_pts, orig_idx


class PartialSAModuleP3DQueryBall(nn.Module):
    def __init__(
        self,
        npoint: int,
        nsample: int,
        mlp_channels: list[int],
        radius: float | None = None
    ):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        # … shared-MLP 定义同前 …

    def forward(self, xyz: torch.Tensor, mask: torch.Tensor):
        device = xyz.device
        N, M, _ = xyz.shape

        # 1. FPS 采样
        fps_out, idx = sample_farthest_from_pts4(xyz, K=self.npoint)
        new_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

        # —— 用 Ball Query 代替 knn_points —— #
        # ball_query 要求输入 (src_points, query_points, radius, max_nn)
        # 返回 group_idx: (N, npoint, nsample)
        assert self.radius is not None, "Ball Query 需要指定 radius"
        group_idx = ball_query(
            xyz,          # 所有点，shape (N, M, 3)
            new_xyz,      # 查询点，shape (N, npoint, 3)
            self.radius,  # 半径 R
            self.nsample  # 最多采样 nsample 个
        )
        # group_idx 里若该中心半径内点 < nsample，会自动 pad 为 0

        # 2. 采集邻域并中心化
        grouped = torch.gather(
            xyz.unsqueeze(1).expand(-1, self.npoint, -1, 3),
            2,
            group_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # (N, npoint, nsample, 3)
        grouped = grouped - new_xyz.unsqueeze(2)

        # 3. shared-MLP + max-pool
        x = grouped.permute(0, 3, 1, 2).to(device)  # (N, 3, npoint, nsample)
        x = self.mlp(x)                            # (N, C_out, npoint, nsample)
        new_feat = x.max(dim=3)[0]                 # (N, C_out, npoint)

        return new_xyz, new_feat


# class PartialSAModuleP3DKNN(nn.Module):
#     """
#     单尺度 Set Abstraction 模块，支持可选的 radius 过滤。
#     1. FPS 采样 npoint 个中心点
#     2. kNN 查询 nsample 个邻居
#     3. （可选）按 radius 丢弃超阈值的邻居
#     4. 局部 shared-MLP + max-pool → 每个中心生成一个 token
#     """

#     def __init__(
#         self,
#         npoint: int,
#         nsample: int,
#         mlp_channels: list[int],
#         radius: float | None = None
#     ):
#         """
#         Args:
#             npoint:      中心点数
#             nsample:     每个中心的邻居数
#             mlp_channels: shared-MLP 各层通道列表，例如 [64,64,128]
#             radius:      如果不为 None，则仅保留距离 <= radius 的邻居，其它置零
#         """
#         super().__init__()
#         self.npoint = npoint
#         self.nsample = nsample
#         self.radius = radius

#         # 构建 shared-MLP：Conv2d 实现 1×1 卷积
#         layers = []
#         last_c = 3
#         for out_c in mlp_channels:
#             layers += [
#                 nn.Conv2d(last_c, out_c, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout2d(0.1),   # 加个 Dropout2d
#             ]
#             last_c = out_c
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, xyz: torch.Tensor, mask: torch.Tensor):
#         """
#         Args:
#             xyz:  (N, M, 3)   点云坐标，padding 用 (0,0,0)
#             mask: (N, M)      True=有效点, False=padding
#         Returns:
#             new_xyz:  (N, npoint, 3)
#             new_feat: (N, C_out, npoint)
#         """
#         device = xyz.device
#         N, M, _ = xyz.shape

#         # 1. FPS 采样 npoint 点
#         fps_out, idx = sample_farthest_from_pts4(xyz, K=self.npoint)
#         new_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

#         # 2. kNN 查询 nsample 邻居
#         knn = knn_points(new_xyz, xyz, K=self.nsample, return_nn=True)
#         # knn.idx: (N, npoint, nsample)
#         # knn.dists: (N, npoint, nsample, 1)  — 距离的平方
#         group_idx = knn.idx

#         # 3. 采集邻域并中心化
#         grouped = torch.gather(
#             xyz.unsqueeze(1).expand(-1, self.npoint, -1, 3),
#             2, group_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
#         )  # (N, npoint, nsample, 3)
#         grouped = grouped - new_xyz.unsqueeze(2)

#         # 4. （可选）radius 过滤：把超出 radius 的邻居置零
#         if self.radius is not None:
#             dists2 = knn.dists.squeeze(-1)         # (N, npoint, nsample)
#             mask_r = dists2 <= (self.radius ** 2)  # (N, npoint, nsample)
#             grouped = grouped * mask_r.unsqueeze(-1)  # 超出半径的坐标成 0

#         # 5. shared-MLP + max-pool
#         # 转成 (N, 3, npoint, nsample) 供 Conv2d
#         x = grouped.permute(0, 3, 1, 2).to(device)
#         x = self.mlp(x)                  # (N, C_out, npoint, nsample)
#         new_feat = x.max(dim=3)[0]       # (N, C_out, npoint)

#         return new_xyz, new_feat


class SAModuleP3D_MSG(nn.Module):
    def __init__(self, npoint: int, radii: list[float], nsamples: list[int], mlp_channels: list[list[int]], d_model: int):
        """
        radii:    各个尺度的半径列表，比如 [0.1, 0.2, 0.4]
        nsamples: 各个尺度对应的邻居数列表，比如 [16, 32, 64]
        mlp_channels: 各尺度的 shared-MLP 通道列表列表，
                      比如 [[32,32,64], [64,64,128], [64,96,128]]
        """
        super().__init__()
        self.npoint = npoint
        assert len(radii) == len(nsamples) == len(mlp_channels)
        self.scales = len(radii)

        # 每个尺度一套 SA + CF + MLP
        self.sa_modules = nn.ModuleList()
        self.cf_filters = nn.ModuleList()
        for r, k, mlp_ch in zip(radii, nsamples, mlp_channels):
            self.sa_modules.append(
                # 用 sample_farthest_points + knn_points，但传入不同半径、邻居数
                PartialSAModuleP3DQueryBall(self.npoint, k, mlp_ch, radius=r)
            )
            self.cf_filters.append(ConfidenceFilter())

        # 最后把所有尺度输出的 feature concat 再映到下一层
        last_C = sum(ch[-1] for ch in mlp_channels)
        self.to_dmodel = nn.Conv1d(last_C, d_model, kernel_size=1)

    def forward(self, xyz: torch.Tensor, mask: torch.Tensor):
        feats = []
        for sa, cf in zip(self.sa_modules, self.cf_filters):
            new_xyz, new_feat = sa(xyz, mask)
            new_feat = cf(new_feat)
            feats.append(new_feat)  # (N, C_i, npoint)
        # 多尺度 concat
        feat_concat = torch.cat(feats, dim=1)  # (N, sum(C_i), npoint)
        tokens = self.to_dmodel(feat_concat).permute(0, 2, 1)
        return new_xyz, tokens


# class SAModuleP3D(nn.Module):
#     """
#     使用 PyTorch3D ops 实现的一层 Set Abstraction:
#     1. FPS 采样 npoint 个中心点
#     2. kNN 查询 nsample 个邻居
#     3. 局部 shared-MLP + max-pool → 为每个中心生成一个 token
#     """

#     def __init__(self, npoint: int, nsample: int, mlp_channels: list[int]):
#         """
#         Args:
#             npoint:      中心点个数 M'
#             nsample:     每个中心的邻居点数 K
#             mlp_channels: 局部 MLP 各层通道列表，例如 [64,64,128]
#         """
#         super().__init__()
#         self.npoint = npoint
#         self.nsample = nsample

#         # 构建 shared-MLP：使用 Conv2d 实现对每个邻域点的 1×1 卷积
#         layers = []
#         last_c = 3
#         for out_c in mlp_channels:
#             layers += [
#                 nn.Conv2d(last_c, out_c, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True),
#             ]
#             last_c = out_c
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, xyz: torch.Tensor, mask: torch.Tensor):
#         """
#         Args:
#             xyz:  (N, M, 3)   点云坐标，(0,0,0) 表示 padding
#             mask: (N, M)      True=有效点, False=padding
#         Returns:
#             new_xyz:  (N, npoint, 3)
#             new_feat: (N, C_out, npoint)
#         """
#         device = xyz.device
#         N, M, _ = xyz.shape

#         # 1. FPS 采样
#         fps_out, idx = sample_farthest_points(xyz, K=self.npoint)
#         # fps_out.idx: (N, npoint) 采样点索引；fps_out.packed_points(): (N*npoint, 3)                               # (N, npoint)
#         new_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (N, npoint, 3)

#         # 2. kNN 查询邻居
#         # knn_points 返回一个命名元组，.idx 是 (N, npoint, nsample)
#         knn = knn_points(new_xyz, xyz, K=self.nsample, return_nn=True)
#         group_idx = knn.idx                                    # (N, npoint, nsample)

#         # 3. 采集邻域坐标并中心化
#         grouped = torch.gather(
#             xyz.unsqueeze(1).expand(-1, self.npoint, -1, 3),
#             2, group_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
#         )  # (N, npoint, nsample, 3)
#         grouped = grouped - new_xyz.unsqueeze(2)               # (N, npoint, nsample, 3)

#         # 4. shared-MLP + 局部 max-pool
#         x = grouped.permute(0, 3, 1, 2).to(device)              # (N, 3, npoint, nsample)
#         x = self.mlp(x)                                        # (N, C_out, npoint, nsample)
#         new_feat = x.max(dim=3)[0]                             # (N, C_out, npoint)

#         return new_xyz, new_feat

# -----------------------
# 整体 Pipeline：输出 Transformer 输入 pc_latent
# -----------------------


class PointNet2FeatureExtractorP3D(nn.Module):
    """
    从 (N, M, 3) 点云到 Transformer token 序列 (N, M_L, d_model)
    完全基于 PyTorch3D 的采样和邻域查询。
    """

    def __init__(self, sa_cfgs: list[dict], d_model: int):
        """
        Args:
            sa_cfgs: List of dict，每个 dict 包含：
                     {'npoint':int, 'nsample':int, 'mlp_channels':[...]}
            d_model: Transformer token 维度
        """
        super().__init__()
        self.sa_modules = nn.ModuleList()
        self.cf_filters = nn.ModuleList()
        for cfg in sa_cfgs:
            self.sa_modules.append(
                SAModuleP3D_MSG(cfg['npoint'], cfg['nsample'], cfg['mlp_channels'])
            )
            self.cf_filters.append(ConfidenceFilter())

        last_C = sa_cfgs[-1]['mlp_channels'][-1]
        self.to_dmodel = nn.Conv1d(last_C, d_model, kernel_size=1)

    def forward(self, pts: torch.Tensor, eps: float = 1e-2):
        """
        Args:
            pts: (N, M, 3) 点云输入，(0,0,0) 表示 padding
        Returns:
            tokens: (N, M_L, d_model)
        """
        device = pts.device
        N, M, _ = pts.shape

        # 构建有效点掩码
        mask = (pts.norm(dim=-1) > eps).to(device)

        # 分层 SA + ConfidenceFilter
        xyz, feat = pts, None
        for sa, cf in zip(self.sa_modules, self.cf_filters):
            xyz, feat = sa(xyz, mask)   # xyz:(N, M_i,3); feat:(N, C_i, M_i)
            feat = cf(feat)             # (N, C_i, M_i)
            mask = torch.ones(N, xyz.size(1), device=device, dtype=torch.bool)

        # 映射到 d_model 并 permute
        feat = self.to_dmodel(feat)    # (N, d_model, M_L)
        tokens = feat.permute(0, 2, 1)  # (N, M_L, d_model)

        return tokens


# # -----------------------
# # 示例用法
# # -----------------------
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # 示例参数
#     N, M, d_model = 4, 1024, 128
#     sa_cfgs = [
#         {'npoint': 256, 'nsample': 32, 'mlp_channels': [64, 64, 128]},
#         {'npoint': 64, 'nsample': 32, 'mlp_channels': [128, 128, 256]},
#     ]

#     model = PointNet2FeatureExtractorP3D(sa_cfgs, d_model).to(device)

#     # 随机示例点云 + padding
#     pts = torch.randn(N, M, 3, device=device)
#     pts[pts.abs().sum(dim=-1) < 0.1] = 0

#     tokens = model(pts)  # → (N, 64, 128)
#     print(tokens.shape, tokens.device)
