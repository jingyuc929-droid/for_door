import torch
from typing import Optional


def grid_pattern(length, width, resolution, device):

    # define grid pattern
    x = torch.arange(start=-length / 2, end=length / 2 + 1.0e-9, step=resolution, device=device)
    y = torch.arange(start=-width / 2, end=width / 2 + 1.0e-9, step=resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    # store into ray starts
    num_rays = grid_x.numel()
    grid = torch.zeros(num_rays, 3, device=device)
    grid[:, 0] = grid_x.flatten()
    grid[:, 1] = grid_y.flatten()

    return grid


def get_3x3_grid(
    grid: torch.Tensor,
    length: float,
    width: float,
    resolution: float,
    x0: torch.Tensor,  # shape (B,)
    y0: torch.Tensor,  # shape (B,)
    device=None,
) -> torch.Tensor:
    """
    grid: (N,3) 由 meshgrid.flatten() 得到
    x0, y0: (B,) 要查询的一批坐标
    返回: (B,9,3)，每个样本的 3x3 邻域点
    """

    # 1. 重构一维坐标轴
    x = torch.arange(-length / 2, length / 2 + 1e-9, resolution, device=device)
    y = torch.arange(-width / 2, width / 2 + 1e-9, resolution, device=device)
    N_x, N_y = x.numel(), y.numel()

    # 2. 计算批量最近网格索引
    idx_x = torch.round((x0 + length / 2) / resolution).long().clamp(0, N_x - 1)  # (B,)
    idx_y = torch.round((y0 + width / 2) / resolution).long().clamp(0, N_y - 1)  # (B,)

    # 3. 准备 3x3 偏移表 (9,2)
    offsets = torch.tensor([
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ], device=device, dtype=torch.long)  # (9,2)

    # 4. 生成 (B,9,2) 的邻域索引
    center_idx = torch.stack([idx_x, idx_y], dim=1)          # (B,2)
    neigh_idx = center_idx.unsqueeze(1) + offsets.unsqueeze(0)  # (B,9,2)
    neigh_idx[..., 0].clamp_(0, N_x - 1)
    neigh_idx[..., 1].clamp_(0, N_y - 1)

    # 5. 转为扁平索引 (B,9)
    flat_idx = neigh_idx[..., 1] * N_x + neigh_idx[..., 0]     # (B,9)

    # 6. 从 grid 中一次性拉取所有点
    flat_idx_flat = flat_idx.reshape(-1)                     # (B*9,)
    pts_flat = grid[flat_idx_flat]                           # (B*9,3)
    pts = pts_flat.view(-1, 9, 3)                            # (B,9,3)

    return pts


def extract_3x3_patches(
    points: torch.Tensor,                       # (N,4,3)
    length: float,
    width: float,
    resolution: float,
    grid: Optional[torch.Tensor] = None,        # (nx*ny,3)，不传就自动生成
):
    """
    给定一批 (B,4,3) 坐标，以及对应的批量 grid (B, nx*ny, 3)，
    返回每个点的 3×3 邻域格点坐标：
      patches: (B, 4, 9, 3)
    """
    device = points.device
    B = points.shape[0]

    # ——— 1. 网格尺寸 ———
    nx = int(round(length / resolution)) + 1
    ny = int(round(width / resolution)) + 1

    # ——— 2. 将坐标映射到最邻近格点的 (i, j) 索引 ———
    i = torch.round((points[..., 0] + length / 2) / resolution).long().clamp(0, nx - 1)  # (B,4)
    j = torch.round((points[..., 1] + width / 2) / resolution).long().clamp(0, ny - 1)  # (B,4)

    # ——— 3. 定义 3×3 偏移 ———
    offsets = torch.tensor(
        [[-1, -1], [-1, 0], [-1, 1],
         [0, -1], [0, 0], [0, 1],
         [1, -1], [1, 0], [1, 1]],
        device=device, dtype=torch.long  # (9,2)
    )

    # ——— 4. 生成 (B,4,9,2) 的邻域索引 ———
    base_idx = torch.stack((i, j), dim=-1).unsqueeze(-2)  # (B,4,1,2)
    neigh_idx = base_idx + offsets.view(1, 1, 9, 2)         # (B,4,9,2)
    neigh_i = neigh_idx[..., 0].clamp(0, nx - 1)         # (B,4,9)
    neigh_j = neigh_idx[..., 1].clamp(0, ny - 1)         # (B,4,9)

    # ——— 5. 转为线性索引 ———
    idx_lin = neigh_i * ny + neigh_j                     # (B,4,9)

    # ——— 6. 按 batch 从 grid 中取坐标 ———
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(idx_lin)  # (B,4,9)
    patches = grid[batch_idx, idx_lin].reshape(B, -1, 3)                   # (B,4*9,3)

    return patches
